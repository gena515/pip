import os
import platform
import sys
import sysconfig
import types
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from pip._vendor.packaging.tags import _manylinux, _musllinux

from pip._internal.utils import compatibility_tags

ManylinuxModule = Callable[[pytest.MonkeyPatch], types.ModuleType]


@pytest.fixture
def manylinux_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setattr(_manylinux, "_get_glibc_version", lambda *args: (2, 20))
    module_name = "_manylinux"
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.mark.parametrize(
    "version_info, expected",
    [
        ((2,), "2"),
        ((2, 8), "28"),
        ((3,), "3"),
        ((3, 6), "36"),
        # Test a tuple of length 3.
        ((3, 6, 5), "36"),
        # Test a 2-digit minor version.
        ((3, 10), "310"),
    ],
)
def test_version_info_to_nodot(version_info: Tuple[int], expected: str) -> None:
    actual = compatibility_tags.version_info_to_nodot(version_info)
    assert actual == expected


class Testcompatibility_tags:
    def mock_get_config_var(self, **kwd: str) -> Callable[[str], Any]:
        """
        Patch sysconfig.get_config_var for arbitrary keys.
        """
        get_config_var = sysconfig.get_config_var

        def _mock_get_config_var(var: str) -> Any:
            if var in kwd:
                return kwd[var]
            return get_config_var(var)

        return _mock_get_config_var

    def test_no_hyphen_tag(self) -> None:
        """
        Test that no tag contains a hyphen.
        """
        import pip._internal.utils.compatibility_tags

        mock_gcf = self.mock_get_config_var(SOABI="cpython-35m-darwin")

        with patch("sysconfig.get_config_var", mock_gcf):
            supported = pip._internal.utils.compatibility_tags.get_supported()

        for tag in supported:
            assert "-" not in tag.interpreter
            assert "-" not in tag.abi
            assert "-" not in tag.platform


class TestManylinuxTags:
    def teardown_method(self) -> None:
        _manylinux._get_glibc_version.cache_clear()

    @pytest.mark.parametrize(
        "manylinux,expected,glibc_ver",
        [
            (
                "manylinux_2_12_x86_64",
                [
                    "manylinux_2_12_x86_64",
                    "manylinux2010_x86_64",
                    "manylinux_2_11_x86_64",
                    "manylinux_2_10_x86_64",
                    "manylinux_2_9_x86_64",
                    "manylinux_2_8_x86_64",
                    "manylinux_2_7_x86_64",
                    "manylinux_2_6_x86_64",
                    "manylinux_2_5_x86_64",
                    "manylinux1_x86_64",
                ],
                "2.12",
            ),
            (
                "manylinux_2_17_x86_64",
                [
                    "manylinux_2_17_x86_64",
                    "manylinux2014_x86_64",
                    "manylinux_2_16_x86_64",
                    "manylinux_2_15_x86_64",
                    "manylinux_2_14_x86_64",
                    "manylinux_2_13_x86_64",
                    "manylinux_2_12_x86_64",
                    "manylinux2010_x86_64",
                    "manylinux_2_11_x86_64",
                    "manylinux_2_10_x86_64",
                    "manylinux_2_9_x86_64",
                    "manylinux_2_8_x86_64",
                    "manylinux_2_7_x86_64",
                    "manylinux_2_6_x86_64",
                    "manylinux_2_5_x86_64",
                    "manylinux1_x86_64",
                ],
                "2.17",
            ),
            (
                "manylinux_2_5_x86_64",
                [
                    "manylinux_2_5_x86_64",
                    "manylinux1_x86_64",
                ],
                "2.5",
            ),
            (
                "manylinux_2_17_s390x",
                [
                    "manylinux_2_17_s390x",
                    "manylinux2014_s390x",
                ],
                "2.17",
            ),
            (
                "manylinux_2_17_x86_64",
                [
                    "manylinux_2_12_x86_64",
                    "manylinux2010_x86_64",
                    "manylinux_2_11_x86_64",
                    "manylinux_2_10_x86_64",
                    "manylinux_2_9_x86_64",
                    "manylinux_2_8_x86_64",
                    "manylinux_2_7_x86_64",
                    "manylinux_2_6_x86_64",
                    "manylinux_2_5_x86_64",
                    "manylinux1_x86_64",
                ],
                "2.12",
            ),
        ],
    )
    def test_manylinux(
        self,
        monkeypatch: pytest.MonkeyPatch,
        manylinux: str,
        expected: List[str],
        glibc_ver: str,
    ) -> None:
        *_, arch = manylinux.split("_", 3)
        monkeypatch.setattr(sysconfig, "get_platform", lambda: f"linux_{arch}")
        monkeypatch.setattr(platform, "machine", lambda: arch)
        monkeypatch.setattr(
            os, "confstr", lambda x: f"glibc {glibc_ver}", raising=False
        )
        groups: Dict[Tuple[str, str], List[str]] = {}
        supported = compatibility_tags.get_supported(platforms=[manylinux])
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        for arches in groups.values():
            if "any" in arches:
                continue
            assert arches == expected


class TestManylinuxCompatibleTags:
    @pytest.mark.parametrize(
        "machine, major, minor, tf", [("x86_64", 2, 20, False), ("s390x", 2, 22, True)]
    )
    def test_use_manylinux_compatible(
        self,
        monkeypatch: pytest.MonkeyPatch,
        manylinux_module: ManylinuxModule,
        machine: str,
        major: int,
        minor: int,
        tf: bool,
    ) -> None:
        def manylinux_compatible(tag_major: int, tag_minor: int, tag_arch: str) -> bool:
            if tag_major == 2 and tag_minor == 22:
                return tag_arch == "s390x"
            return False

        monkeypatch.setattr(
            _manylinux,
            "_get_glibc_version",
            lambda: (major, minor),
        )
        monkeypatch.setattr(
            manylinux_module,
            "manylinux_compatible",
            manylinux_compatible,
            raising=False,
        )
        groups: Dict[Tuple[str, str], List[str]] = {}
        manylinux = f"manylinux_{major}_{minor}_{machine}"
        supported = compatibility_tags.get_supported(platforms=[manylinux])
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        if tf:
            expected = [f"manylinux_2_22_{machine}"]
        else:
            expected = [f"manylinux_{major}_{minor}_{machine}"]
        for arches in groups.values():
            if "any" in arches:
                continue
            assert arches == expected

    def test_linux_use_manylinux_compatible_none(
        self, monkeypatch: pytest.MonkeyPatch, manylinux_module: ManylinuxModule
    ) -> None:
        def manylinux_compatible(
            tag_major: int, tag_minor: int, tag_arch: str
        ) -> Optional[bool]:
            if tag_major == 2 and tag_minor < 25:
                return False
            return None

        monkeypatch.setattr(_manylinux, "_get_glibc_version", lambda: (2, 30))
        monkeypatch.setattr(sysconfig, "get_platform", lambda: "linux_x86_64")
        monkeypatch.setattr(
            manylinux_module,
            "manylinux_compatible",
            manylinux_compatible,
            raising=False,
        )

        groups: Dict[Tuple[str, str], List[str]] = {}
        supported = compatibility_tags.get_supported(
            platforms=["manylinux_2_30_x86_64"]
        )
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        expected = [
            "manylinux_2_30_x86_64",
            "manylinux_2_29_x86_64",
            "manylinux_2_28_x86_64",
            "manylinux_2_27_x86_64",
            "manylinux_2_26_x86_64",
            "manylinux_2_25_x86_64",
        ]
        for arches in groups.values():
            if "any" in arches:
                continue
            assert arches == expected


class TestManylinux2010Tags:
    @pytest.mark.parametrize(
        "manylinux2010,manylinux1",
        [
            ("manylinux2010_x86_64", "manylinux1_x86_64"),
            ("manylinux2010_i686", "manylinux1_i686"),
        ],
    )
    def test_manylinux2010_implies_manylinux1(
        self, manylinux2010: str, manylinux1: str
    ) -> None:
        """
        Specifying manylinux2010 implies manylinux1.
        """
        groups: Dict[Tuple[str, str], List[str]] = {}
        supported = compatibility_tags.get_supported(platforms=[manylinux2010])
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        for arches in groups.values():
            if arches == ["any"]:
                continue
            assert arches[:2] == [manylinux2010, manylinux1]


class TestManylinux2014Tags:
    @pytest.mark.parametrize(
        "manylinuxA,manylinuxB",
        [
            ("manylinux2014_x86_64", ["manylinux2010_x86_64", "manylinux1_x86_64"]),
            ("manylinux2014_i686", ["manylinux2010_i686", "manylinux1_i686"]),
        ],
    )
    def test_manylinuxA_implies_manylinuxB(
        self, manylinuxA: str, manylinuxB: List[str]
    ) -> None:
        """
        Specifying manylinux2014 implies manylinux2010/manylinux1.
        """
        groups: Dict[Tuple[str, str], List[str]] = {}
        supported = compatibility_tags.get_supported(platforms=[manylinuxA])
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        expected_arches = [manylinuxA]
        expected_arches.extend(manylinuxB)
        for arches in groups.values():
            if arches == ["any"]:
                continue
            assert arches[:3] == expected_arches


class TestMusllinuxTags:
    @pytest.mark.parametrize(
        "musllinux,arch,musl_ver",
        [
            ("musllinux_1_4", "x86_64", (1, 4)),
            ("musllinux_1_4", "i686", (1, 2)),
        ],
    )
    def test_musllinux(
        self,
        monkeypatch: pytest.MonkeyPatch,
        musllinux: str,
        arch: str,
        musl_ver: Tuple[int, int],
    ) -> None:
        monkeypatch.setattr(
            _musllinux,
            "_get_musl_version",
            lambda _: _musllinux._MuslVersion(*musl_ver),
        )
        monkeypatch.setattr(sysconfig, "get_platform", lambda: f"linux_{arch}")
        groups: Dict[Tuple[str, str], List[str]] = {}
        supported = compatibility_tags.get_supported(platforms=[f"{musllinux}_{arch}"])
        for tag in supported:
            groups.setdefault((tag.interpreter, tag.abi), []).append(tag.platform)

        expected = [
            f"musllinux_{musl_ver[0]}_{minor}_{arch}"
            for minor in range(musl_ver[1], -1, -1)
        ]
        for arches in groups.values():
            if "any" in arches:
                continue
            assert arches == expected
