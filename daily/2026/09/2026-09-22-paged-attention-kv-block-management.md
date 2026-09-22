# 2026-09-22：PagedAttention 与 KV 块管理——映射、共享、写时复制和释放

## 今日目标

上一篇 [连续批处理与请求调度](2026-09-21-continuous-batching-request-scheduling.md)把 KV 容量视为位置槽位。今天进一步讨论真实系统为什么按块管理 KV，以及共享之后如何避免错误覆盖和提前释放。

完成后应能回答：

1. 逻辑连续的序列，为什么不必使用物理连续的 KV？
2. 块大小如何影响尾部浪费和管理成本？
3. 共享块为什么需要引用计数和写时复制？
4. 前缀缓存命中为什么不仅取决于当前块的 token？
5. 如何验证分配失败不会破坏已有请求？

本篇代码只用整数列表代表块内容，没有实现 attention kernel、真实 KV 张量或 GPU 分配器。它演示活跃请求之间的显式分叉，不实现自动前缀查找。

---

## 1. PagedAttention 解决的是什么问题

序列长度动态增长时，若为每个请求预留很大的连续 KV 区域，会浪费容量；若要求扩展后仍连续，也增加管理难度。

[PagedAttention 论文](https://arxiv.org/abs/2309.06180)借鉴虚拟内存分页思想，通过块级映射组织 KV，使一个请求的 KV 可以存放在不连续的物理块中。

逻辑块表示序列中的顺序，物理块表示存储池中的位置。attention 执行必须按照映射找到正确数据；分页不是删除历史 token，也不是自动缩小注意力上下文。

## 2. 逻辑位置如何找到物理位置

设块大小为 B，token 位置 t 从零开始：

```math
j=\left\lfloor\frac{t}{B}\right\rfloor,\qquad
o=t\bmod B,\qquad
p=\mathrm{block\_table}[j]
```

j 是逻辑块编号，o 是块内偏移，p 是物理块编号。物理块顺序不必与逻辑顺序相同。

例如 B=4，逻辑块表为 [7, 2, 9]，位置 5 对应物理块 2 的偏移 1。不能把物理地址顺序当作文本顺序。

不同层、KV 组或设备可能有不同布局；这里把一个物理块抽象成包含相应缓存状态的单位。

## 3. 块取整与尾部浪费

对于一个不共享、采用普通全注意力缓存的序列，长度 T 需要：

```math
N_{\mathrm{blocks}}=\left\lceil\frac{T}{B}\right\rceil,
\qquad
W=B\left\lceil\frac{T}{B}\right\rceil-T
```

T=5、B=4 时需要两个块，尾部有 3 个未用槽位。分页减少对大段连续空间的要求，但不意味着没有内部碎片。

小块可能减少尾部浪费，却增加块表、分配和访存管理成本。大块减少管理单元数，却可能降低短前缀复用粒度。具体最优值需要结合 kernel 和工作负载测量。

多请求共享时，物理占用应按唯一物理块统计，不能简单把各请求逻辑块数量相加。

## 4. 共享只读状态与写时复制

两个请求拥有相同的已计算前缀时，可以引用同一物理块。只读访问不需要复制，但一方继续写入共享的未满尾块，会影响另一方。

写时复制（copy-on-write，COW）的做法是：

1. 检查尾块是否共享。
2. 先成功分配新块，并复制有效内容。
3. 将当前请求的尾块映射切换到新块。
4. 减少旧块引用计数，再写入新位置。

如果尾块已经满了，追加通常分配下一个块，不需要复制满块。代码仍需遵循实际执行引擎的数据依赖。

## 5. 引用计数归零意味着什么

在今天的简化池中：

- 每份请求块表引用使计数加一。
- 请求结束时，对其每个块减一。
- 计数归零立即释放内容并回到空闲池。

但自动前缀缓存可以保留没有活跃请求引用的块内容，以供后续复用。因此“活跃引用为零”“可被淘汰”和“内容已经无效”是不同状态。

真正重用物理块存储新内容之前，需要使旧的缓存索引失效，避免后续命中旧前缀却读到新内容。本文代码选择立即释放，不实现缓存驻留或淘汰策略。

## 6. 前缀缓存键不只是当前块 token

同样的一段 token，放在不同前文之后，深层 KV 可能不同。因此缓存身份必须关联前缀上下文，以及会影响 KV 的执行条件，例如模型、适配器和输入表示。

vLLM 的 [Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)说明了父块哈希、块内 token 与附加信息参与缓存标识，也提供 cache_salt 隔离复用范围。具体字段和行为依版本而定。

文本相似不是缓存相同；token 相同也不足以跨不同模型任意共享。多模态内容同样不能只用表面占位符来识别。

哈希是索引机制，不是访问控制或模型正确性的证明。租户隔离和敏感数据处理仍需要独立设计。

## 7. 教学实现：显式分叉的块池

以下约定便于验证：

- 单线程，无异步 GPU 读写。
- 一个请求的块表内不重复引用同一个块。
- fork 共享已有块；append 对共享的未满尾块执行 COW。
- 容量不足抛出 MemoryError，并保持此次 append 前的状态。
- 只保存整数作为示意内容，不计算真实 KV。

```python
from collections import Counter
from copy import deepcopy


class BlockPool:
    def __init__(self, capacity, block_size):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("invalid capacity")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("invalid block size")
        self.capacity = capacity
        self.block_size = block_size
        self.free = set(range(capacity))
        self.data, self.refs, self.tables = {}, {}, {}

    def create(self, rid):
        if not isinstance(rid, str) or not rid or rid in self.tables:
            raise ValueError("invalid or duplicate request")
        self.tables[rid] = []

    def allocate(self, content):
        if not self.free:
            raise MemoryError("no free blocks")
        pid = min(self.free)
        self.free.remove(pid)
        self.data[pid] = list(content)
        self.refs[pid] = 1
        return pid

    def fork(self, source, target):
        if source not in self.tables:
            raise ValueError("unknown source")
        self.create(target)
        self.tables[target] = self.tables[source].copy()
        for pid in self.tables[target]:
            self.refs[pid] += 1

    def append(self, rid, token):
        if rid not in self.tables or type(token) is not int:
            raise ValueError("invalid request or token")
        table = self.tables[rid]
        if not table or len(self.data[table[-1]]) == self.block_size:
            new = self.allocate([token])  # Fail before changing the table.
            table.append(new)
            return
        old = table[-1]
        if self.refs[old] > 1:
            new = self.allocate(self.data[old])  # COW allocation first.
            self.refs[old] -= 1
            table[-1] = new
        self.data[table[-1]].append(token)

    def release(self, rid):
        if rid not in self.tables:
            raise ValueError("unknown or already released request")
        for pid in self.tables.pop(rid):
            self.refs[pid] -= 1
            if self.refs[pid] == 0:
                del self.refs[pid]
                del self.data[pid]
                self.free.add(pid)

    def read(self, rid):
        return [token for pid in self.tables[rid] for token in self.data[pid]]

    def check(self):
        counts = Counter(pid for table in self.tables.values() for pid in table)
        assert dict(counts) == self.refs
        assert set(self.refs) == set(self.data)
        assert self.free.isdisjoint(self.data)
        assert self.free | set(self.data) == set(range(self.capacity))
        for table in self.tables.values():
            assert len(table) == len(set(table))
            for index, pid in enumerate(table):
                assert 1 <= len(self.data[pid]) <= self.block_size
                if index < len(table) - 1:
                    assert len(self.data[pid]) == self.block_size
```

allocate 是内部辅助方法，教学调用只通过 append 使用。这里的失败原子性仅针对显式的块容量不足，不覆盖 Python 进程内存耗尽、线程竞争、设备错误或崩溃恢复。

## 8. 验证：共享、分叉、释放与容量不足

以下代码接在上一段之后执行。

```python
pool = BlockPool(4, 4)
pool.create("a")
for token in range(5):
    pool.append("a", token)
pool.fork("a", "b")
pool.check()
assert len(pool.data) == 2  # Four logical references, two physical blocks.

pool.append("b", 99)
pool.check()
assert pool.read("a") == [0, 1, 2, 3, 4]
assert pool.read("b") == [0, 1, 2, 3, 4, 99]
assert pool.tables["a"][0] == pool.tables["b"][0]
assert pool.tables["a"][-1] != pool.tables["b"][-1]
assert len(pool.data) == 3

pool.release("a")
pool.check()
assert len(pool.data) == 2
assert pool.read("b") == [0, 1, 2, 3, 4, 99]
pool.release("b")
pool.check()
assert len(pool.free) == 4

# Full shared tail: append allocates a new block, without copying the full one.
full = BlockPool(2, 2)
full.create("a")
full.append("a", 1)
full.append("a", 2)
full.fork("a", "b")
full.append("b", 3)
full.check()
assert full.tables["a"][0] == full.tables["b"][0]
assert full.read("a") == [1, 2]
assert full.read("b") == [1, 2, 3]

# Shared partial tail, but no space for COW: neither request may be altered.
tight = BlockPool(1, 4)
tight.create("a")
tight.append("a", 7)
tight.fork("a", "b")
before = deepcopy(tight.__dict__)
try:
    tight.append("b", 8)
except MemoryError:
    pass
else:
    raise AssertionError("expected capacity failure")
assert tight.__dict__ == before
tight.check()
tight.release("a")
tight.append("b", 8)  # Tail is now uniquely owned; no allocation needed.
tight.check()
assert tight.read("b") == [7, 8]
tight.release("b")
tight.check()

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: tight.release("b"))
reject(lambda: pool.fork("missing", "new"))
reject(lambda: BlockPool(0, 4))
assert (5 // 4, 5 % 4) == (1, 1)
assert ((5 + 4 - 1) // 4) * 4 - 5 == 3
print("shared=2 blocks, after COW=3 blocks, after release=0 blocks")
print("OOM preserved state; exclusive tail append succeeded.")
print("All KV block-management checks passed.")
```

这些测试证明教学实现中的分叉不会覆盖另一请求，并检查引用计数与池容量守恒。它们不验证 attention 数值等价性。

## 9. 从今天的块池回到昨天的调度器

[昨天的调度例子](2026-09-21-continuous-batching-request-scheduling.md)按新增位置扣容量。今天可以看到：

- 独占尾块尚未满：追加位置可能不需要新物理块。
- 尾块已满：追加需要新块。
- 尾块未满但共享：追加可能需要 COW 新块。
- 请求结束：只有最后一个活跃引用消失才可按本例释放。

因此接纳计划必须询问块管理器的实际需求，而不是只计算“新增 token 数”。如果先修改调度状态再发现分配失败，还需要一致的回滚策略。

## 10. 正确性验收先于性能比较

真实实现应在小输入上将分页路径与可信的非分页路径比较，使用相同权重、位置、mask 和精度，检查输出误差是否处于明确容差内。

进一步测试：

- 跨块边界增长、不同长度和空前缀。
- 同时完成、取消和异常请求。
- 分叉后不同分支继续生成。
- 容量耗尽、抢占及块复用。
- GPU 仍在读取时不能过早复用对应存储。

浮点误差和调度变化可能影响采样，不能仅通过“最终文字差不多”证明 kernel 正确。性能测试再报告有效 KV 利用率、缓存命中、复制量、TTFT 和尾延迟。

## 11. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
shared=2 blocks, after COW=3 blocks, after release=0 blocks
OOM preserved state; exclusive tail append succeeded.
All KV block-management checks passed.
```

已检查 UTF-8、Markdown 围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

今天只实现活跃分支共享、引用计数和写时复制。没有实现哈希查找、前缀缓存驻留、LRU 淘汰、多租户隔离、GPU kernel 或跨设备 KV 传输。

示例中的“释放”表示回到教学池，不意味着真实服务会把显存归还操作系统。实际框架经常保留预分配存储池供后续请求使用。

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。
- [vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)。
- [前篇：连续批处理与请求调度](2026-09-21-continuous-batching-request-scheduling.md)。

## 今日总结

1. 逻辑连续不要求物理连续，但读取必须遵守块表。
2. 分页仍有尾部浪费和元数据开销。
3. 共享状态写入前需要明确所有权，必要时执行 COW。
4. 无活跃引用不必然表示缓存内容已失效。
5. 容量失败和请求释放都必须保持映射与引用计数一致。

## 下次衔接建议

继续学习推理量化：区分权重量化与 KV Cache 量化，理解 scale、零点、误差以及容量、速度和质量之间的权衡。
