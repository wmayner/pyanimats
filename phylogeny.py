#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# phylogeny.py

"""Provides objects and methods for handling phylogenetic information."""

from collections import UserList


class Phylogeny(UserList):
    """A population of animats.

    Behaves like a normal list, but allows pickling the population with the
    phylogenetic tree of their lineages.

    Time complexity is ``O(n)`` for insertion and deletion, where ``n`` is the
    size of the animat's lineage.

    Note that changing the parent reference of an element after creation will
    result in undefined behavior, and inserting an animat with circular parent
    references will cause an infinite loop.
    """

    def __init__(self, animats):
        self.data = animats
        self._lookup = {}
        for animat in animats:
            self._insert(animat)

    def _insert(self, animat):
        """Inserts an animat and its lineage into the lookup table.

        Note: this will hang for circular parent references.
        """
        while animat is not None:
            # Increment refcount if it's already there.
            if animat._id in self._lookup:
                self._lookup[animat._id][1] += 1
            # Otherwise insert with refcount of 1.
            else:
                self._lookup[animat._id] = [animat, 1]
            animat = animat.parent

    def _remove(self, animat):
        """Removes an animat and its lineage from the lookup table."""
        while animat is not None:
            # Delete if refcount is 1.
            if self._lookup[animat._id][1] == 1:
                del self._lookup[animat._id]
            # Otherwise decrement refcount.
            else:
                self._lookup[animat._id][1] -= 1
            animat = animat.parent

    def __setstate__(self, state):
        self.__dict__.update(state)
        for animat_ref in self._lookup.values():
            if animat_ref[0].parent is not None:
                animat_ref[0].parent = self._lookup[animat_ref[0].parent][0]

    def __setitem__(self, position, animat):
        super().__setitem__(position, animat)
        self._insert(animat)

    def __delitem__(self, position):
        self._remove(self[position])
        super().__delitem__(position)

    def append(self, animat):
        super().append(animat)
        self._insert(animat._id)

    def insert(self, position, animat):
        super().insert(position, animat)
        self._insert(animat)

    def pop(self, i=-1):
        self._remove(self[i])
        return super().pop(i)

    def remove(self, animat):
        super().remove(animat)
        self._remove(animat)

    def extend(self, animats):
        super().extend(animats)
        for animat in animats:
            self._insert(animat)
