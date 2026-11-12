import pytest

from mneme.pipeline import (
    split_top_level,
    flatten_passes,
    parse_llvm_pass_options,
    PipelineManager,
    AbstractPass,
)

# ------------------------------------------------------------
# split_top_level
# ------------------------------------------------------------

def test_split_top_level_simple():
    s = "a,b,c"
    assert split_top_level(s) == ["a", "b", "c"]


def test_split_top_level_nested():
    s = "a(b,c), d, e(f(g,h),i)"
    assert split_top_level(s) == [
        "a(b,c)",
        "d",
        "e(f(g,h),i)",
    ]


# ------------------------------------------------------------
# flatten_passes
# ------------------------------------------------------------

def test_flatten_passes_basic():
    s = "a(b,c),d"
    out = flatten_passes(s)
    assert out == ["b", "c", "d"]


def test_flatten_passes_deep():
    s = "aa(bb(cc,dd),ee),ff"
    out = flatten_passes(s)
    assert out == ["cc", "dd", "ee", "ff"]


# ------------------------------------------------------------
# parse_llvm_pass_options
# ------------------------------------------------------------

def test_parse_llvm_pass_options_toggle():
    name, opts = parse_llvm_pass_options("myPass<foo;no-foo>")
    assert name == "myPass"
    assert "foo" in opts
    assert opts["foo"].is_toggle()


def test_parse_llvm_pass_options_range():
    name, opts = parse_llvm_pass_options("myPass<threshold=N>")
    assert "threshold" in opts
    assert opts["threshold"].is_range()


def test_parse_llvm_pass_options_setting_list():
    name, opts = parse_llvm_pass_options("myPass<optA;optB;optC>")
    assert name == "myPass"
    assert "myPass" in opts
    assert opts["myPass"].is_setting()
    assert opts["myPass"].options == ["optA", "optB", "optC"]


# ------------------------------------------------------------
# PipelineManager basic construction
# ------------------------------------------------------------

def test_pipeline_manager_constructs():
    pm = PipelineManager()
    assert isinstance(pm._passes, list)
    assert len(pm._passes) > 10


# ------------------------------------------------------------
# to_string
# ------------------------------------------------------------

def test_to_string_names_the_pass():
    pm = PipelineManager()
    concrete = AbstractPass.ConcretePass(pm._kw_passes["mem2reg"], [])
    assert "mem2reg" in pm.to_string([concrete])


# ------------------------------------------------------------
# Random pipeline generation
# ------------------------------------------------------------

def test_generate_deterministic():
    pm = PipelineManager()
    p1 = pm.generate(samples=2, mean_passes=3, std=1, seed=123)
    p2 = pm.generate(samples=2, mean_passes=3, std=1, seed=123)
    assert p1[0][0].__str__() == p2[0][0].__str__()


def test_generate_length_clamped_min():
    pm = PipelineManager()
    p = pm._generate_random_pipeline(mean_passes=0.1, std_passes=0.1)
    assert len(p) == 1


def test_generate_length_clamped_max():
    pm = PipelineManager()
    max_len = len(pm._passes)
    p = pm._generate_random_pipeline(mean_passes=max_len * 10, std_passes=1)
    assert len(p) == max_len

