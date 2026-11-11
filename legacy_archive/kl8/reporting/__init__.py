"""全渠道三维报表渲染模块（契约 R48-d、§3.13）。

包含：
- 控制台与 TXT 精要六段式决策报告 (render_text_report, save_text_report)
- 微信/移动端精巧文案 (render_wechat, save_wechat)
- 商业级 6-Sheet Excel 报表 (render_excel, ExcelStyle)
- 三类交付件唯一数据源上下文 (ReportContext)
"""

from kl8.reporting.excel import ExcelStyle, render_excel
from kl8.reporting.text_report import ReportContext, render_text_report, save_text_report
from kl8.reporting.wechat import render_wechat, save_wechat

__all__ = [
    "ReportContext",
    "render_text_report",
    "save_text_report",
    "render_wechat",
    "save_wechat",
    "ExcelStyle",
    "render_excel",
]
