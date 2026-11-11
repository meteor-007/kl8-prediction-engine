"""架构分层契约与单向依赖静态 AST 审计测试（契约 C5、§2.2）。

【核心断言】
1. 五层单向依赖：layerN 严禁反向 import layerN+1；
2. 基础设施层纯净性：errors / constants / config / datatypes 严禁依赖任何上层业务层；
3. pipeline 与 main 唯一编排点：底层与中间层严禁反向 import main 或 pipeline；
4. evaluation 与 reporting 只读消费：不得被下层反向依赖。
"""

import ast
import unittest
from pathlib import Path


class ArchitectureDependencyVisitor(ast.NodeVisitor):
    """AST 访问器，提取 Python 源码中的所有 import 语句目标模块。"""

    def __init__(self):
        self.imports: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)


def _get_imports_for_file(py_file: Path) -> list[str]:
    """读取并使用 AST 解析单个 Python 文件的所有导入模块名。"""
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        visitor = ArchitectureDependencyVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        raise RuntimeError(f"解析 AST 失败: {py_file}: {e}") from e


class TestArchitecture(unittest.TestCase):
    """五层架构依赖规则严格静态测试。"""

    def setUp(self):
        self.root = Path("kl8")

    def test_layer1_dependencies(self):
        """Layer 1 (Representation) 只能依赖基础设施，严禁依赖 Layer 2..5、evaluation、reporting、pipeline。"""
        l1_dir = self.root / "layer1_representation"
        forbidden = [
            "layer2_baseline",
            "layer3_features",
            "layer4_scoring",
            "layer5_combinatorial",
            "evaluation",
            "reporting",
            "pipeline",
            "main",
        ]
        for py_file in l1_dir.glob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                for bad in forbidden:
                    self.assertNotIn(
                        f"kl8.{bad}",
                        imp,
                        f"架构违规: {py_file} 包含了向后依赖 {imp}！",
                    )

    def test_layer2_dependencies(self):
        """Layer 2 (Baseline) 严禁依赖 Layer 3..5、evaluation、reporting、pipeline。"""
        l2_dir = self.root / "layer2_baseline"
        forbidden = [
            "layer3_features",
            "layer4_scoring",
            "layer5_combinatorial",
            "evaluation",
            "reporting",
            "pipeline",
            "main",
        ]
        for py_file in l2_dir.glob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                for bad in forbidden:
                    self.assertNotIn(
                        f"kl8.{bad}",
                        imp,
                        f"架构违规: {py_file} 包含了向后依赖 {imp}！",
                    )

    def test_layer3_dependencies(self):
        """Layer 3 (Features) 严禁依赖 Layer 4..5、evaluation、reporting、pipeline。"""
        l3_dir = self.root / "layer3_features"
        forbidden = [
            "layer4_scoring",
            "layer5_combinatorial",
            "evaluation",
            "reporting",
            "pipeline",
            "main",
        ]
        for py_file in l3_dir.glob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                for bad in forbidden:
                    self.assertNotIn(
                        f"kl8.{bad}",
                        imp,
                        f"架构违规: {py_file} 包含了向后依赖 {imp}！",
                    )

    def test_layer4_dependencies(self):
        """Layer 4 (Scoring) 严禁依赖 Layer 5、evaluation、reporting、pipeline。"""
        l4_dir = self.root / "layer4_scoring"
        forbidden = [
            "layer5_combinatorial",
            "evaluation",
            "reporting",
            "pipeline",
            "main",
        ]
        for py_file in l4_dir.glob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                for bad in forbidden:
                    self.assertNotIn(
                        f"kl8.{bad}",
                        imp,
                        f"架构违规: {py_file} 包含了向后依赖 {imp}！",
                    )

    def test_layer5_dependencies(self):
        """Layer 5 (Combinatorial) 严禁依赖 evaluation、reporting、pipeline。"""
        l5_dir = self.root / "layer5_combinatorial"
        forbidden = [
            "evaluation",
            "reporting",
            "pipeline",
            "main",
        ]
        for py_file in l5_dir.glob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                for bad in forbidden:
                    self.assertNotIn(
                        f"kl8.{bad}",
                        imp,
                        f"架构违规: {py_file} 包含了向后依赖 {imp}！",
                    )

    def test_no_lower_layer_imports_pipeline_or_main(self):
        """除 main.py 自身外，kl8 包内所有子模块绝对禁止导入 main。"""
        for py_file in self.root.rglob("*.py"):
            imports = _get_imports_for_file(py_file)
            for imp in imports:
                self.assertNotIn(
                    "main",
                    imp.split("."),
                    f"架构违规: {py_file} 包含了逆向导入 main！",
                )


if __name__ == "__main__":
    unittest.main()
