# expect: WA
# A recipe's creation number is the number of recipes that exist when it is added.
import itertools


class _Recipe:
    __slots__ = ("rid", "seq", "owner", "editors", "versions")

    def __init__(self, rid, seq, owner, name, ingredients):
        self.rid, self.seq, self.owner = rid, seq, owner
        self.editors = set()
        self.versions = [(name, tuple(ingredients))]

    @property
    def name(self):
        return self.versions[-1][0]

    @property
    def ingredients(self):
        return self.versions[-1][1]

    def view(self):
        return {"id": self.rid, "name": self.name, "ingredients": list(self.ingredients)}


class RecipeManager:
    def __init__(self):
        self._recipes = {}
        self._by_key = {}
        self._users = set()
        self._seq = itertools.count()

    # ---- helpers
    def _taken_by_other(self, name, rid):
        holder = self._by_key.get(name.casefold())
        return holder is not None and holder != rid

    def _set_current(self, rec, name, ingredients):
        del self._by_key[rec.name.casefold()]
        rec.versions.append((name, tuple(ingredients)))
        self._by_key[name.casefold()] = rec.rid

    def _may_edit(self, user_id, rec):
        return user_id == rec.owner or user_id in rec.editors

    def _ordered(self, keep=lambda rec: True):
        recs = sorted(self._recipes.values(), key=lambda rec: (len(rec.ingredients), rec.seq))
        return [rec.view() for rec in recs if keep(rec)]

    # ---- Level 1 (final forms of add/delete are Level 3's)
    def add_recipe(self, recipe_id, name, ingredients, owner_id):
        if recipe_id in self._recipes or name.casefold() in self._by_key or owner_id not in self._users:
            return False
        self._recipes[recipe_id] = _Recipe(recipe_id, len(self._recipes), owner_id, name, ingredients)
        self._by_key[name.casefold()] = recipe_id
        return True

    def get_recipe(self, recipe_id):
        rec = self._recipes.get(recipe_id)
        return None if rec is None else rec.view()

    def update_recipe(self, recipe_id, name, ingredients):
        rec = self._recipes.get(recipe_id)
        if rec is None or self._taken_by_other(name, recipe_id):
            return False
        self._set_current(rec, name, ingredients)
        return True

    def delete_recipe(self, user_id, recipe_id):
        rec = self._recipes.get(recipe_id)
        if rec is None or rec.owner != user_id:
            return False
        del self._recipes[recipe_id]
        del self._by_key[rec.name.casefold()]
        return True

    # ---- Level 2
    def list_recipes(self):
        return self._ordered()

    def find_by_ingredient(self, ingredient):
        want = ingredient.casefold()
        return self._ordered(lambda rec: any(i.casefold() == want for i in rec.ingredients))

    def search_by_name_prefix(self, prefix):
        p = prefix.casefold()
        return self._ordered(lambda rec: rec.name.casefold().startswith(p))

    # ---- Level 3
    def add_user(self, user_id, username):
        if user_id in self._users:
            return False
        self._users.add(user_id)
        return True

    def grant_editor(self, owner_id, recipe_id, user_id):
        rec = self._recipes.get(recipe_id)
        if rec is None or rec.owner != owner_id or user_id not in self._users:
            return False
        rec.editors.add(user_id)
        return True

    def edit_recipe(self, user_id, recipe_id, name, ingredients):
        rec = self._recipes.get(recipe_id)
        if rec is None or not self._may_edit(user_id, rec) or self._taken_by_other(name, recipe_id):
            return False
        self._set_current(rec, name, ingredients)
        return True

    # ---- Level 4
    def get_recipe_history(self, recipe_id):
        rec = self._recipes.get(recipe_id)
        if rec is None:
            return None
        return [{"version": k, "name": n, "ingredients": list(ing)} for k, (n, ing) in enumerate(rec.versions, 1)]

    def get_version(self, recipe_id, version):
        rec = self._recipes.get(recipe_id)
        if rec is None or not 1 <= version <= len(rec.versions):
            return None
        n, ing = rec.versions[version - 1]
        return {"version": version, "name": n, "ingredients": list(ing)}

    def rollback_recipe(self, user_id, recipe_id, version):
        rec = self._recipes.get(recipe_id)
        if rec is None or not self._may_edit(user_id, rec) or not 1 <= version <= len(rec.versions):
            return False
        n, ing = rec.versions[version - 1]
        if self._taken_by_other(n, recipe_id):
            return False
        self._set_current(rec, n, ing)
        return True
