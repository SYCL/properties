# sycl_khr_properties — Minimal C++20 Implementation

A standalone, header-only reference implementation of the SYCL compile-time
properties extension proposed in
[KhronosGroup/SYCL-Docs#980](https://github.com/KhronosGroup/SYCL-Docs/pull/980).

This KHR introduces a `properties` class that supersedes `sycl::property_list`,
enabling compile-time property propagation so that:
- APIs can detect property errors at compilation time rather than runtime.
- Properties can influence codegen (e.g. kernel specialization).

This repository explores six design iterations for the key mechanism.

## Tony Table — Design comparison

### Defining a compile-time `int` property

| v1 (spec) | v2 (default param) | v3 (`template_key`) | v4 (`ct_value` NTTP) | v5 (parameter pack) | v6 (`auto` NTTP + `no_value`) |
|---|---|---|---|---|---|
| `struct prop_key : constant_key_base {};` | | | | | |
| `template<int V>` | `template<int V = 0>` | `template<int V>` | `template<ct_value<int> V = {}>` | `template<int...> struct prop;` | `template<auto V = no_value{}>` |
| `struct prop_impl { ... };` | `struct prop { ... };` | `struct prop { ... };` | `struct prop { ... };` | `template<> struct prop<> { ... };` | `struct prop_t { ... };` |
| `template<int V>` | | | | `template<int V>` | `template<auto V = no_value{}>` |
| `constexpr prop_impl<V> prop{};` | | | | `struct prop<V> { ... };` | `constexpr prop_t<V> prop{};` |

### Constructing a property value

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| `prop<42>` | `prop<42>{}` | `prop<42>{}` | `prop<42>{}` | `prop<42>{}` | `prop<42>` |

### Querying a property

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| `has_property<prop_key>()` | `has_property<prop<>>()` | `has_property<template_key<prop>>()` | `has_property<prop<>>()` | `has_property<prop<>>()` | `has_property<prop<>>()` |
| `get_property<prop_key>()` | `get_property<prop<>>()` | `get_property<template_key<prop>>()` | `get_property<prop<>>()` | `get_property<prop<>>()` | `get_property<prop<>>()` |

### Is `prop<0>` a valid property (not the key)?

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| N/A (no `prop<0>`) | **No** — `prop<0>` == `prop<>` | Yes | Yes (NTTP bool) | Yes | Yes |

### Is `enable<false>` valid for a boolean property?

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| N/A | **No** — `enable<false>` == `enable<>` | Yes | Yes (NTTP bool) | Yes | Yes |

### Key mechanism

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| Separate `_key` type | Default instantiation (`prop<>` == `prop<0>`) | `template_key<prop>` wrapper | `ct_value<T>` NTTP with bool discriminant | Empty parameter pack (`prop<>` vs `prop<V>`) | `no_value` sentinel **type** via `auto` NTTP |

### Separate key type needed?

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| Yes (`prop_key`) | No | No (but `template_key` wrapper) | No | No | No |

### Variable template (no `{}`)?

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| Yes | No | No | No | No | Yes |

### Extra boilerplate per property

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| Separate key struct | None | None | None | Two specializations | Internal `_t` struct |

### Portability concern

| v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| None | None | `template<auto...>` matching | `ct_value` as NTTP (C++20) | None | `auto` NTTP (C++20) |

## Property taxonomy

The extension defines three property kinds, distinguished by how their values
are conveyed:

| Kind | Value carrier | Key | Storage in `properties` |
|------|--------------|-----|------------------------|
| **Runtime** | Constructor args → member variables | The property type itself | Stored in internal tuple |
| **Compile-time** | Template parameters → `static constexpr` members | Varies by design (see table above) | Not stored (stateless, reconstructed) |
| **Hybrid** | Template params + constructor args | Same as compile-time | Stored in internal tuple |

## v6 — Recommended design

v6 combines the best properties: variable template syntax, no value
reservation, no runtime booleans, and clean query syntax using `prop<>`.

### Defining properties

```cpp
// Runtime property — struct, self-keyed.
struct in_order : detail::runtime_property<in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

// Compile-time property — auto NTTP with no_value sentinel.
// prop_t is internal; prop is the user-facing variable template.
template <auto V = no_value{}>
struct prop_t : detail::constant_property_base {
  using __detail_key_t = prop_t<>;
  static constexpr bool has_value = !detail::is_sentinel_v<V>;
  static constexpr auto value = V;
};
template <auto V = no_value{}>
inline constexpr prop_t<V> prop{};

// Hybrid property — auto NTTP + runtime constructor arg.
template <auto V1 = no_value{}>
struct prop3 : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;
  constexpr prop3(int v2) : value2(v2) {}
  static constexpr auto value1 = V1;
  int value2;
};
```

### Using properties

```cpp
using namespace sycl::khr6;

// Construction — variable templates, no {} needed.
constexpr properties plist{prop<42>, in_order{true}};

// Query — prop<> is a constexpr value passed as NTTP.
static_assert(decltype(plist)::has_property<prop<>>());
static_assert(decltype(plist)::has_property<in_order>());

// Compile-time get — NTTP overload.
static_assert(decltype(plist)::get_property<prop<>>().value == 42);

// Runtime get — type parameter overload.
static_assert(plist.get_property<in_order>().value == true);

// prop<0> is a valid property, distinct from the key prop<>.
constexpr properties p{prop<0>};
static_assert(decltype(p)::get_property<prop<>>().value == 0);
```

## Trait machinery

Six traits classify properties, keys, and property lists:

| Trait | Detected via |
|-------|-------------|
| `is_property<T>` | `is_base_of` against `{runtime,constant,hybrid}_property_base` |
| `is_property_key<T>` | Runtime: always. Compile-time/hybrid: `T == T::__detail_key_t` |
| `is_property_key_compile_time<T>` | `is_base_of<constant_property_base, T>` + self-keyed |
| `is_property_for<T, Class>` | Delegates to `is_property_key_for<T::__detail_key_t, Class>` |
| `is_property_key_for<T, Class>` | Explicit specialization per property x class pair |
| `is_property_list_for<T, Class>` | Partial specialization on `properties<...>`, folds `is_property_for` |

Only `is_property_key_for` needs a per-property specialization; all other traits
are derived automatically from the tag-base inheritance.

## Architecture

```
include/sycl/
  khr/                        -- v1 (spec-faithful, separate key types)
  khr2/                       -- v2 (default param as key, sentinel caveat)
  khr3/                       -- v3 (template_key wrapper)
  khr4/                       -- v4 (ct_value NTTP with bool)
  khr5/                       -- v5 (parameter pack)
  khr6/                       -- v6 (auto NTTP + no_value sentinel type)
    detail/
      property_bases.hpp      -- Tag bases, no_value, runtime_property CRTP
      property_meta.hpp       -- find_property, filter, duplicate detection
    properties.hpp            -- properties class, 6 traits, NTTP overloads
    queue_properties.hpp      -- enable_profiling, in_order
test/
  test_properties.cpp         -- v1 tests
  test_properties_v2.cpp      -- v2 tests
  ...
  test_properties_v6.cpp      -- v6 tests
```

### Key implementation details

- **Property-to-key mapping**: Every property type exposes
  `using __detail_key_t = <key>;`. Runtime properties are their own key.
  Compile-time properties use the sentinel instantiation (with `no_value`).

- **Storage**: Only properties with runtime state are stored (in a
  `std::tuple`). Compile-time properties are stateless and reconstructed via
  default construction in `get_property()`.

- **`has_property`/`get_property` dual overloads**: Both accept
  `template<typename>` (for runtime property keys like `enable_profiling`) and
  `template<auto>` (for variable-template keys like `prop<>`). The NTTP
  overloads delegate to the type-parameter versions via `decltype(Key)`.

- **Duplicate key detection**: `static_assert` in the constructor using a fold
  expression that counts key occurrences.

### Known deviations from the spec

- The spec requires `properties` to be trivially copyable when all contained
  runtime properties are trivially copyable. This implementation uses
  `std::tuple` for storage, which is not trivially copyable. A production
  implementation would use a custom aggregate with `[[no_unique_address]]`.

- No real SYCL runtime — `queue_tag` is used as a stub `Class` parameter for
  trait specializations.

## Building

Requires a C++20 compiler (GCC 12+, Clang 15+, MSVC 19.34+).

```bash
cmake -B build
cmake --build build
./build/test_properties_v6
```

## References

- [SYCL-Docs PR #980](https://github.com/KhronosGroup/SYCL-Docs/pull/980) — KHR specification draft
- [intel/llvm#21368](https://github.com/intel/llvm/pull/21368) — Proof-of-concept implementation
