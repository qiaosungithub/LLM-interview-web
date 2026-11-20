# expect: WA
# Editors are kept per recipe_id outside the recipe, so a re-created recipe inherits them.
def _key(name: str) -> str:
    return name.casefold()


class _Recipe:
    def __init__(self, recipe_id: str, name: str, ingredients: list[str]):
        self.id = recipe_id
        self.name = name
        self.ingredients = list(ingredients)
        self.owner_id: str | None = None                  # set by Level 3's add_recipe
        self.editors: set[str] = set()                     # grown by Level 3's grant_editor (membership only)
        self.versions: list[tuple[str, list[str]]] = [(name, list(ingredients))]   # read from Level 4 on

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "ingredients": list(self.ingredients)}


class RecipeManager:
    def __init__(self) -> None:
        self._recipes: dict[str, _Recipe] = {}      # recipe_id -> _Recipe, in creation order
        self._name_index: dict[str, str] = {}         # lookup key -> the recipe_id currently holding it

    def _name_taken(self, name: str, exclude_id: str | None = None) -> bool:
        holder = self._name_index.get(_key(name))
        return holder is not None and holder != exclude_id

    def _rename(self, recipe: _Recipe, name: str) -> None:
        del self._name_index[_key(recipe.name)]
        recipe.name = name
        self._name_index[_key(name)] = recipe.id

    def add_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        if recipe_id in self._recipes or self._name_taken(name):
            return False
        self._recipes[recipe_id] = _Recipe(recipe_id, name, ingredients)
        self._name_index[_key(name)] = recipe_id
        return True

    def get_recipe(self, recipe_id: str) -> dict | None:
        recipe = self._recipes.get(recipe_id)
        return recipe.public() if recipe is not None else None

    def update_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or self._name_taken(name, exclude_id=recipe_id):
            return False
        self._rename(recipe, name)
        recipe.ingredients = list(ingredients)
        return True

    def delete_recipe(self, recipe_id: str) -> bool:
        recipe = self._recipes.pop(recipe_id, None)
        if recipe is None:
            return False
        del self._name_index[_key(recipe.name)]
        return True

class RecipeManager(RecipeManager):
    def list_recipes(self) -> list[dict]:
        return self._sorted(self._recipes.values())

    def find_by_ingredient(self, ingredient: str) -> list[dict]:
        key = _key(ingredient)
        return self._sorted(r for r in self._recipes.values() if any(_key(i) == key for i in r.ingredients))

    def search_by_name_prefix(self, prefix: str) -> list[dict]:
        key = _key(prefix)
        return self._sorted(r for r in self._recipes.values() if _key(r.name).startswith(key))

    @staticmethod
    def _sorted(recipes) -> list[dict]:
        return [r.public() for r in sorted(recipes, key=lambda r: len(r.ingredients))]

class RecipeManager(RecipeManager):
    def __init__(self) -> None:
        super().__init__()
        self._users: set[str] = set()
        self._editors: dict[str, set[str]] = {}   # membership only, like _Recipe.editors -- never iterated for output

    def add_user(self, user_id: str, username: str) -> bool:
        if user_id in self._users:
            return False
        self._users.add(user_id)
        return True

    def add_recipe(self, recipe_id: str, name: str, ingredients: list[str], owner_id: str) -> bool:
        if owner_id not in self._users:
            return False
        if not super().add_recipe(recipe_id, name, ingredients):
            return False
        self._recipes[recipe_id].owner_id = owner_id
        return True

    def _authorized(self, user_id: str, recipe: _Recipe | None, allow_editor: bool) -> bool:
        if recipe is None:
            return False
        return user_id == recipe.owner_id or (allow_editor and user_id in self._editors.get(recipe.id, set()))

    def grant_editor(self, owner_id: str, recipe_id: str, user_id: str) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or owner_id != recipe.owner_id or user_id not in self._users:
            return False
        self._editors.setdefault(recipe_id, set()).add(user_id)
        return True

    def edit_recipe(self, user_id: str, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        recipe = self._recipes.get(recipe_id)
        if not self._authorized(user_id, recipe, allow_editor=True) or self._name_taken(name, exclude_id=recipe_id):
            return False
        self._rename(recipe, name)
        recipe.ingredients = list(ingredients)
        return True

    def delete_recipe(self, user_id: str, recipe_id: str) -> bool:
        recipe = self._recipes.get(recipe_id)
        if not self._authorized(user_id, recipe, allow_editor=False):
            return False
        return super().delete_recipe(recipe_id)

class RecipeManager(RecipeManager):
    def edit_recipe(self, user_id: str, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        if not super().edit_recipe(user_id, recipe_id, name, ingredients):
            return False
        recipe = self._recipes[recipe_id]
        recipe.versions.append((recipe.name, list(recipe.ingredients)))
        return True

    def get_recipe_history(self, recipe_id: str) -> list[dict] | None:
        recipe = self._recipes.get(recipe_id)
        if recipe is None:
            return None
        return [{"version": i, "name": n, "ingredients": list(ing)}
                for i, (n, ing) in enumerate(recipe.versions, start=1)]

    def get_version(self, recipe_id: str, version: int) -> dict | None:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or not 1 <= version <= len(recipe.versions):
            return None
        name, ingredients = recipe.versions[version - 1]
        return {"version": version, "name": name, "ingredients": list(ingredients)}

    def rollback_recipe(self, user_id: str, recipe_id: str, version: int) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or not 1 <= version <= len(recipe.versions):
            return False
        name, ingredients = recipe.versions[version - 1]
        return self.edit_recipe(user_id, recipe_id, name, ingredients)
