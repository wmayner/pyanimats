#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# phylogeny.py

"""Provides objects and methods for handling phylogenetic information."""

from collections import UserList


class Phylogeny(UserList):
    """A population of animats.

    Behaves like a normal list, but allows pickling the population with the
    phylogenetic tree of their lineages.

    Time complexity is ``O(n/step)`` for insertion and deletion, where ``n`` is
    the size of the animat's lineage and ``step`` is the generational interval.

    Note that changing the parent reference of an element after creation will
    result in undefined behavior, and inserting an animat with circular parent
    references will cause an infinite loop.
    """

    def __init__(self, animats, step=1):
        self.data = animats
        self._lookup = {}
        self.step = step
        for animat in animats:
            self._insert(animat)

    def _incref(self, animat, parent):
        # Increment refcount if it's already there.
        if animat._id in self._lookup:
            self._lookup[animat._id][1] += 1
        # Otherwise insert with refcount of 1.
        else:
            parent_id = parent._id if parent is not None else None
            self._lookup[animat._id] = [animat, 1, parent_id]

    def _decref(self, animat):
        # Delete if refcount is 1.
        if self._lookup[animat._id][1] == 1:
            del self._lookup[animat._id]
        # Otherwise decrement refcount.
        else:
            self._lookup[animat._id][1] -= 1

    def _insert(self, animat):
        """Inserts an animat and its lineage into the lookup table."""
        lineage = animat.lineage(step=self.step)
        child = next(lineage)
        for ancestor in lineage:
            parent = ancestor
            self._incref(child, parent)
            child = parent

    def _lineage(self, animat):
        """Returns the lineage of the animat as it exists in the phylogeny."""
        ancestor_id = animat._id
        while ancestor_id is not None:
            ancestor_ref = self._lookup[ancestor_id]
            yield ancestor_ref[0]
            ancestor_id = ancestor_ref[2]

    def _remove(self, animat):
        """Removes an animat and its lineage from the lookup table."""
        for ancestor in self._lineage(animat):
            self._decref(ancestor)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for animat_ref in self._lookup.values():
            animat, parent_id = animat_ref[0], animat_ref[2]
            animat.parent = (self._lookup[parent_id][0]
                             if parent_id is not None else None)

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
