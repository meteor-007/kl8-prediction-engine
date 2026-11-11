"""覆盖设计标准旋转矩阵查表与重验测试（契约 R38）。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kl8.layer5_combinatorial.covering import lookup_table, verify_guarantee


class TestCoveringTables(unittest.TestCase):
    """覆盖表文件完整性与穷举核验测试。"""

    def test_tables_verified(self) -> None:
        """测试 covering_tables.json 中的每个经典区组表 100% 穷举核验通过（契约 R38）。"""
        table_path = Path("kl8/layer5_combinatorial/covering_tables.json")
        self.assertTrue(table_path.exists(), "覆盖表文件必须存在")

        with open(table_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertTrue(len(data) >= 3, "覆盖表应至少包含经典三组以上参数")

        for key, tickets in data.items():
            parts = [int(p) for p in key.split(",")]
            v, k, t, m = parts[0], parts[1], parts[2], parts[3]
            res = verify_guarantee(tickets, v, k, t, m)
            self.assertTrue(
                res.passed,
                f"覆盖表 {key} 未通过穷举保底验证: achieved_t={res.achieved_t} < t={t}",
            )


if __name__ == "__main__":
    unittest.main()
