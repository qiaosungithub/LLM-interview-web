class RecipeManager:
    def __init__(self) -> None:
        """Creates an empty manager: no recipes and no users yet."""

    def add_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        """Level 1: creates a recipe; False, without effect, if recipe_id or name's lookup key is taken.
        (Level 3 replaces it with add_recipe(recipe_id, name, ingredients, owner_id).)"""
        raise NotImplementedError

    def get_recipe(self, recipe_id: str) -> dict | None:
        """Level 1: returns {"id": recipe_id, "name": ..., "ingredients": [...]}, or None if recipe_id names no recipe."""
        raise NotImplementedError

    def update_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        """Level 1: replaces recipe_id's name and ingredients; False, without effect, if it fails."""
        raise NotImplementedError

    def delete_recipe(self, recipe_id: str) -> bool:
        """Level 1: removes recipe_id, freeing its id and name; False if it named no recipe.
        (Level 3 replaces it with delete_recipe(user_id, recipe_id).)"""
        raise NotImplementedError

    def list_recipes(self) -> list[dict]:
        """Level 2: every recipe, by number of ingredients ascending, ties in creation order."""
        raise NotImplementedError

    def find_by_ingredient(self, ingredient: str) -> list[dict]:
        """Level 2: every recipe with an ingredient equal to ingredient, case-insensitive, in list_recipes order."""
        raise NotImplementedError

    def search_by_name_prefix(self, prefix: str) -> list[dict]:
        """Level 2: every recipe whose name starts with prefix, case-insensitive, in list_recipes order."""
        raise NotImplementedError

    def add_user(self, user_id: str, username: str) -> bool:
        """Level 3: registers user_id; False, without effect, if it is already registered."""
        raise NotImplementedError

    def grant_editor(self, owner_id: str, recipe_id: str, user_id: str) -> bool:
        """Level 3: the recipe's owner grants user_id permission to edit recipe_id."""
        raise NotImplementedError

    def edit_recipe(self, user_id: str, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        """Level 3: the owner or a granted editor changes recipe_id's name and ingredients."""
        raise NotImplementedError

    def get_recipe_history(self, recipe_id: str) -> list[dict] | None:
        """Level 4: [{"version": 1, "name": ..., "ingredients": [...]}, ...], oldest first; None if no recipe."""
        raise NotImplementedError

    def get_version(self, recipe_id: str, version: int) -> dict | None:
        """Level 4: the {"version", "name", "ingredients"} entry for this version, or None."""
        raise NotImplementedError

    def rollback_recipe(self, user_id: str, recipe_id: str, version: int) -> bool:
        """Level 4: appends a copy of that version as recipe_id's newest version; False, without effect, if it fails."""
        raise NotImplementedError
