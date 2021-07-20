//! The `lang` module contains functionality to assist in bootstrapping the core language.

// Contains the source of the `core` namespace
mod core;
/// Contains the language "prelude", a module for interop that
/// provides primitives available in every program.
pub mod prelude;

pub use self::core::SOURCE as CORE_SOURCE;
