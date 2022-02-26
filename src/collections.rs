use itertools::sorted;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, IntoIterator};
use std::ops::{Deref, DerefMut};

use archery::RcK;
use rpds::map::hash_trie_map::Iter as MapIter;
use rpds::set::hash_trie_set::Iter as SetIter;
use rpds::{HashTrieMap, HashTrieSet};
pub use rpds::{List as PersistentList, Vector as PersistentVector};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistentMap<K: Eq + Hash + Ord, V: Eq + Ord + Hash>(HashTrieMap<K, V>);

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> Deref for PersistentMap<K, V> {
    type Target = HashTrieMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> DerefMut for PersistentMap<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> PartialOrd for PersistentMap<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(sorted(self.0.iter()).cmp(sorted(other.0.iter())))
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> Ord for PersistentMap<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> Hash for PersistentMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.size().hash(state);
        sorted(self.0.iter()).for_each(|binding| binding.hash(state));
    }
}

impl<'a, K: Eq + Hash + Ord, V: Eq + Ord + Hash> IntoIterator for &'a PersistentMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = MapIter<'a, K, V, RcK>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> PersistentMap<K, V> {
    pub fn new() -> Self {
        Self(HashTrieMap::new())
    }
}

impl<K: Eq + Hash + Ord, V: Eq + Ord + Hash> FromIterator<(K, V)> for PersistentMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(into_iter: I) -> Self {
        Self(HashTrieMap::from_iter(into_iter))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistentSet<T: Hash + Eq + Ord>(HashTrieSet<T>);

impl<T: Hash + Eq + Ord> Deref for PersistentSet<T> {
    type Target = HashTrieSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Hash + Eq + Ord> DerefMut for PersistentSet<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Hash + Eq + Ord> PartialOrd for PersistentSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(sorted(self.0.iter()).cmp(sorted(other.0.iter())))
    }
}

impl<T: Hash + Eq + Ord> Ord for PersistentSet<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: Hash + Eq + Ord> Hash for PersistentSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.size().hash(state);
        sorted(self.0.iter()).for_each(|binding| binding.hash(state));
    }
}

impl<'a, T: Hash + Eq + Ord> IntoIterator for &'a PersistentSet<T> {
    type Item = &'a T;
    type IntoIter = SetIter<'a, T, RcK>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T: Eq + Hash + Ord> PersistentSet<T> {
    pub fn new() -> Self {
        Self(HashTrieSet::new())
    }
}

impl<T: Hash + Eq + Ord> FromIterator<T> for PersistentSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(into_iter: I) -> Self {
        Self(HashTrieSet::from_iter(into_iter))
    }
}
