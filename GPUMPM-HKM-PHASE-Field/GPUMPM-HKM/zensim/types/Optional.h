#pragma once
#include <optional>

namespace zs {

  template <typename T> using optional = std::optional<T>;
  using nullopt_t = std::nullopt_t;
  constexpr auto nullopt = std::nullopt;

}  // namespace zs
