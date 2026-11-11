"""全流程流水线与 CLI 端到端单测（契约 R48、§3.14）。"""

import json
import tempfile
import unittest
from pathlib import Path

from kl8.config import AdaptiveConfig, AppConfig
from kl8.pipeline import run_audit, run_backtest, run_daily
from main import main


class TestPipeline(unittest.TestCase):
    """端到端流水线测试集。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.output_dir = self.temp_path / "output"
        self.state_path = self.temp_path / "state" / "adaptive_weights.json"
        self.cfg = AppConfig(
            output_dir=str(self.output_dir),
            history_path="kl8_history.txt",
            adaptive=AdaptiveConfig(state_path=str(self.state_path)),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_temp_config(self) -> Path:
        """写最小 JSON 覆盖配置，供 main CLI 隔离 output/state，避免污染仓库。"""
        config_path = self.temp_path / "test_config.json"
        payload = {
            "history_path": str(Path("kl8_history.txt").resolve()),
            "output_dir": str(self.output_dir),
            "adaptive": {"state_path": str(self.state_path)},
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return config_path

    def test_end_to_end_daily_and_backtest(self):
        """测试日常预测与伴随回测端到端落盘与三维交付物验证（契约 R48）。"""
        res = run_daily(
            self.cfg,
            no_sync=True,
            target_period=2026292,
            target_date="2026-11-02",
            backtest_n=5,
        )

        self.assertEqual(res.target_period, 2026292)
        self.assertEqual(res.target_date, "2026-11-02")
        self.assertEqual(len(res.pool.gold), 4, "契约 A5：main=12 时 n_gold=clip(round(0.30×12),3,5)=4")
        self.assertGreaterEqual(len(res.pool.main), 10)

        # 验证三维交付件全部生成落盘
        self.assertTrue(res.report_paths["text"].exists(), "TXT 精要报表未生成！")
        self.assertTrue(res.report_paths["wechat"].exists(), "微信文案未生成！")
        self.assertTrue(res.report_paths["excel"].exists(), "商业 Excel 报表未生成！")
        self.assertTrue(res.report_paths["state"].exists(), "自适应状态文件未落盘！")

        # 验证独立回测执行
        bt = run_backtest(self.cfg, n=5)
        self.assertEqual(bt.n_sample, 5)
        self.assertIn(5, bt.lift)

    def test_run_audit(self):
        """测试系统级自洽性与防穿越审计。"""
        audit_res = run_audit(self.cfg, n=5)
        self.assertTrue(audit_res["code_audit_clean"], f"发现违规打乱代码: {audit_res['code_violations']}")
        self.assertEqual(audit_res["backtest_samples"], 5)

    def test_main_cli_dispatch(self):
        """测试 main CLI 入口在不同参数下的调度与退出码规范（output/state 重定向到 tmp）。"""
        config_path = self._write_temp_config()

        # 1. 成功预测退出码 0，且交付件全部落在临时目录（不得写入仓库 output/）
        ret_daily = main(
            ["--no-sync", "--backtest", "0", "--period", "2026292", "--config", str(config_path)]
        )
        self.assertEqual(ret_daily, 0)
        self.assertTrue(self.output_dir.is_dir())
        self.assertTrue(list((self.output_dir / "reports").glob("*.txt")))
        self.assertTrue(list((self.output_dir / "wechat").glob("*.txt")))
        self.assertTrue(list((self.output_dir / "excel").glob("*.xlsx")))

        # 2. 审计模式退出码 0
        ret_audit = main(["--mode", "audit", "--backtest", "5", "--config", str(config_path)])
        self.assertEqual(ret_audit, 0)

        # 3. 非法参数应捕获 ConfigError 并返回退出码 1
        ret_bad = main(["--k", "99"])
        self.assertEqual(ret_bad, 1)


if __name__ == "__main__":
    unittest.main()
