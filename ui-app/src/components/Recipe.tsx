import React, { useEffect, useState } from "react";
import { type RecipeData } from "../datatypes";

const Recipe = ({ recipe: data }: { recipe: RecipeData }): JSX.Element => {
  const [showDetails, setShowDetails] = useState(false);

  const toggleDetails = (): void => {
    setShowDetails(!showDetails);
  };

  return (
    <div className="bg-white shadow-md rounded-md overflow-hidden col-span-4 p-2 m-2">
      <div className="bg-gray-200 p-4">
        <h1 className="text-2xl font-semibold">{data.name}</h1>
        <p className="text-sm font-medium text-gray-500">{data.description}</p>
      </div>
      {showDetails && (
        <div className="p-4 space-y-2">
          <div>
            <h2 className="text-lg font-medium">Ingredients</h2>
            <p className="text-gray-500">{data.ingredients}</p>
          </div>
          <div>
            <h2 className="text-lg font-medium">Steps</h2>
            <p className="text-gray-500">{data.steps}</p>
          </div>
          <div>
            <h2 className="text-lg font-medium">Nutrition Information</h2>
            <ul className="text-gray-500">
              <li>Calories: {data.calories}</li>
              <li>Protein: {data.protein}g</li>
              <li>Carbohydrates: {data.carbohydrates}g</li>
              <li>Sugar: {data.sugar}g</li>
              <li>Total Fat: {data.total_fat}g</li>
              <li>Saturated Fat: {data.saturated_fat}g</li>
              <li>Sodium: {data.sodium}mg</li>
              <li>Compound: {data.compound}</li>
            </ul>
          </div>
          <div>
            <h2 className="text-lg font-medium">Reviews</h2>
            <p className="text-gray-500">{data.review}</p>
          </div>
          <div>
            <h2 className="text-lg font-medium">Recipe Details</h2>
            <ul className="text-gray-500">
              <li>Contributor ID: {data.contributor_id}</li>
              <li>User ID: {data.user_id}</li>
              <li>Recipe ID: {data.recipe_id}</li>
              <li>Date Submitted: {data.submitted}</li>
              <li>Date Posted: {data.date}</li>
              <li>Rating: {data.rating}</li>
              <li>Food Types: {data.food_types}</li>
              <li>Number of Ingredients: {data.n_ingredients}</li>
              <li>Number of Steps: {data.n_steps}</li>
              <li>Tags: {data.tags}</li>
              <li>Minutes to Make: {data.minutes}</li>
            </ul>
          </div>
        </div>
      )}
      <div className="p-4 text-right">
        <button
          className="text-blue-500 font-medium focus:outline-none"
          onClick={toggleDetails}
        >
          {showDetails ? "Hide Details" : "Show Details"}
        </button>
      </div>
    </div>
  );
};

export const Recipes = ({
  recipes,
}: {
  recipes: RecipeData[];
}): JSX.Element => {
  const [recipeData, setRecipeData] = useState<RecipeData[]>(recipes);

  useEffect(() => {
    setRecipeData(recipes);
  }, [recipes]);

  return (
    <div className="grid grid-cols-12">
      {recipes.map((recipe, index) => (
        <Recipe recipe={recipe} key={recipe.recipe_id} />
      ))}
    </div>
  );
};
