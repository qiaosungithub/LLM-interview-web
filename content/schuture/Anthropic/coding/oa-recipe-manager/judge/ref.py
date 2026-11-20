"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator. Every check re-scans what it needs; nothing is indexed or cached.

The model has the final (Level 3/4) API. Level 1-2 sequences are written in their Level 1 form and fed to
it through users_form(): one registered owner OWNER owns every recipe, update_recipe becomes that owner's
edit_recipe and delete_recipe(id) becomes that owner's delete_recipe. Seen through Level 1-2's methods the
two forms behave the same."""

OWNER = "u0"


class _Recipe:
    def __init__(self, seq, name, ingredients, owner):
        self.seq = seq                      # creation number: ties in list order are broken by it
        self.name = name
        self.ingredients = list(ingredients)
        self.owner = owner
        self.editors = set()
        self.versions = [(name, list(ingredients))]


class RecipeManager:
    def __init__(self):
        self.recipes = {}       # recipe_id -> _Recipe
        self.users = set()
        self.created = 0

    def _collides(self, name, other_than=None):
        key = name.casefold()
        return any(rid != other_than and r.name.casefold() == key for rid, r in self.recipes.items())

    def _view(self, rid):
        r = self.recipes[rid]
        return {"id": rid, "name": r.name, "ingredients": list(r.ingredients)}

    def _ordered(self, ids):
        ids = sorted(ids, key=lambda rid: (len(self.recipes[rid].ingredients), self.recipes[rid].seq))
        return [self._view(rid) for rid in ids]

    def can_edit(self, uid, rid):
        r = self.recipes.get(rid)
        return r is not None and (uid == r.owner or uid in r.editors)

    # Level 1 (in its Level 3 form)
    def add_recipe(self, rid, name, ingredients, owner):
        if rid in self.recipes or self._collides(name) or owner not in self.users:
            return False
        self.created += 1
        self.recipes[rid] = _Recipe(self.created, name, ingredients, owner)
        return True

    def get_recipe(self, rid):
        return self._view(rid) if rid in self.recipes else None

    # Level 2
    def list_recipes(self):
        return self._ordered(self.recipes)

    def find_by_ingredient(self, ingredient):
        key = ingredient.casefold()
        return self._ordered([rid for rid, r in self.recipes.items()
                              if any(i.casefold() == key for i in r.ingredients)])

    def search_by_name_prefix(self, prefix):
        key = prefix.casefold()
        return self._ordered([rid for rid, r in self.recipes.items() if r.name.casefold().startswith(key)])

    # Level 3
    def add_user(self, uid, username):
        if uid in self.users:
            return False
        self.users.add(uid)
        return True

    def grant_editor(self, owner, rid, uid):
        r = self.recipes.get(rid)
        if r is None or owner != r.owner or uid not in self.users:
            return False
        r.editors.add(uid)
        return True

    def edit_recipe(self, uid, rid, name, ingredients):
        if not self.can_edit(uid, rid) or self._collides(name, rid):
            return False
        r = self.recipes[rid]
        r.name, r.ingredients = name, list(ingredients)
        r.versions.append((name, list(ingredients)))
        return True

    def delete_recipe(self, uid, rid):
        r = self.recipes.get(rid)
        if r is None or uid != r.owner:
            return False
        del self.recipes[rid]
        return True

    # Level 4
    def get_recipe_history(self, rid):
        r = self.recipes.get(rid)
        if r is None:
            return None
        return [{"version": k, "name": n, "ingredients": list(i)} for k, (n, i) in enumerate(r.versions, 1)]

    def get_version(self, rid, version):
        r = self.recipes.get(rid)
        if r is None or not 1 <= version <= len(r.versions):
            return None
        n, i = r.versions[version - 1]
        return {"version": version, "name": n, "ingredients": list(i)}

    def rollback_recipe(self, uid, rid, version):
        r = self.recipes.get(rid)
        if r is None or not 1 <= version <= len(r.versions):
            return False
        n, i = r.versions[version - 1]
        return self.edit_recipe(uid, rid, n, i)


def users_form(method, args):
    """A Level 1 call in the form a Level 3 class takes, owned by OWNER; other calls unchanged."""
    if method == "add_recipe" and len(args) == 3:
        return method, (*args, OWNER)
    if method == "update_recipe":
        return "edit_recipe", (OWNER, *args)
    if method == "delete_recipe" and len(args) == 1:
        return method, (OWNER, *args)
    return method, args


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight
    1: {"add_recipe": 4, "get_recipe": 3, "update_recipe": 4, "delete_recipe": 2},
    2: {"add_recipe": 4, "get_recipe": 1, "update_recipe": 4, "delete_recipe": 2,
        "list_recipes": 3, "find_by_ingredient": 3, "search_by_name_prefix": 3},
    3: {"add_user": 2, "add_recipe": 4, "get_recipe": 1, "grant_editor": 4, "edit_recipe": 5, "delete_recipe": 2,
        "list_recipes": 1, "find_by_ingredient": 1, "search_by_name_prefix": 1},
    4: {"add_user": 2, "add_recipe": 3, "get_recipe": 1, "grant_editor": 3, "edit_recipe": 4, "delete_recipe": 1,
        "list_recipes": 1, "get_recipe_history": 2, "get_version": 3, "rollback_recipe": 5},
}
RIDS = ["r1", "r2", "r3", "r4"]
USERS = ["u1", "u2", "u3", "x"]
# "Straße" and "STRASSE" collide (casefold) but differ under lower(). Prefix and ingredient searches say only
# "case-insensitive", so names with ß are used where no search runs (Level 1) and nowhere else.
NAMES_L1 = ["Congee", "congee", "Congee ", "Toast", "Jook", "Straße", "STRASSE", "strasse", "Miso Soup", "粥"]
NAMES = ["Congee", "congee", "Congee ", "Corn", "Toast", "Jook", "Miso Soup", "miso", "粥"]
INGREDIENTS = ["rice", "Rice", "RICE", "water", "egg", "Egg", "eggplant", "salt", "bread", "ginger"]


def apply(model, level, method, args):
    if level <= 2:
        method, args = users_form(method, args)
    return getattr(model, method)(*args)


def random_ops(r, level, n_ops):
    """A sequence of (method, args) calls, drawn while a model follows along so that the draws often hit
    the interesting cases: colliding and re-cased names, reused ids, owners, editors and strangers."""
    m = RecipeManager()
    if level <= 2:
        m.add_user(OWNER, "owner")
    kinds, weights = zip(*KINDS[level].items())
    names = NAMES_L1 if level == 1 else NAMES
    ops = []

    def variant(s):
        return r.choice([s, s, s.upper(), s.lower(), s.title(), s.swapcase()])

    def pick_rid():
        return r.choice(list(m.recipes)) if m.recipes and r.random() < 0.75 else r.choice(RIDS)

    def pick_name(rid):
        own = m.recipes.get(rid)
        others = [x.name for k, x in m.recipes.items() if k != rid]
        p = r.random()
        if own is not None and p < 0.25:
            return variant(own.name)
        if others and p < 0.55:
            return variant(r.choice(others))
        return r.choice(names)

    def pick_ingredients():
        return [r.choice(INGREDIENTS) for _ in range(r.choice([0, 1, 1, 2, 2, 3, 3, 4]))]

    def pick_user(rid=None, owner=0.4, editor=0.25):
        x, p = m.recipes.get(rid), r.random()
        if x is not None and p < owner:
            return x.owner
        if x is not None and x.editors and p < owner + editor:
            return r.choice(sorted(x.editors))
        if m.users and r.random() < 0.6:
            return r.choice(sorted(m.users))
        return r.choice(USERS)

    for _ in range(n_ops):
        kind = r.choices(kinds, weights)[0]
        if level >= 3 and len(m.users) < 2 and r.random() < 0.7:
            kind = "add_user"
        if kind == "add_user":
            args = (r.choice(USERS), r.choice(["Alice", "Bob", "Alice", ""]))
        elif kind == "add_recipe":
            rid = r.choice(RIDS) if r.random() < 0.7 else pick_rid()
            args = (rid, pick_name(rid), pick_ingredients())
            if level >= 3:
                args += (pick_user(owner=0, editor=0),)
        elif kind in ("update_recipe", "edit_recipe"):
            rid = pick_rid()
            args = (rid, pick_name(rid), pick_ingredients())
            if level >= 3:
                args = (pick_user(rid),) + args
        elif kind == "delete_recipe":
            rid = pick_rid()
            args = (rid,) if level <= 2 else (pick_user(rid, owner=0.45, editor=0.3), rid)
        elif kind == "grant_editor":
            rid = pick_rid()
            args = (pick_user(rid, owner=0.6, editor=0.2), rid, pick_user(rid, owner=0.1, editor=0.2))
        elif kind in ("get_recipe", "get_recipe_history"):
            args = (pick_rid(),)
        elif kind == "list_recipes":
            args = ()
        elif kind == "find_by_ingredient":
            args = (variant(r.choice(INGREDIENTS + ["plant", "eg"])),)
        elif kind == "search_by_name_prefix":
            s = r.choice(names + ["x"])
            args = (variant(s[:r.randint(0, len(s))]),)
        else:   # get_version, rollback_recipe
            rid = pick_rid()
            x = m.recipes.get(rid)
            top = len(x.versions) if x else 2
            version = r.choice([r.randint(1, top), r.randint(1, top), top, 0, top + 1, -1])
            args = (rid, version) if kind == "get_version" else (pick_user(rid, owner=0.5, editor=0.3), rid, version)
        ops.append((kind, args))
        apply(m, level, kind, args)
    return ops
