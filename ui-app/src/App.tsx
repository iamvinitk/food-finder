import React, { useEffect } from "react";
import { Recipes } from "./components/Recipe";
import { type RecipeData } from "./datatypes";
import { useDebounce } from "./useDebounce";

// const data: RecipeData[] = [
//   {
//     calories: 598.6,
//     carbohydrates: 32.0,
//     compound: 0.8591,
//     contributor_id: 37779,
//     date: "2006-11-29",
//     description: "southern living. always served for thanksgiving dinner.",
//     food_types: "Non-veg",
//     ingredients:
//       "canned pumpkin, sweetened condensed milk, eggs, light brown sugar, sugar, ground cinnamon, salt, ground ginger, ground nutmeg, ground cloves, pie crusts",
//     minutes: 79,
//     n_ingredients: 11,
//     n_steps: 7,
//     name: "best ever pumpkin pie",
//     negative: 0.0,
//     neutral: 0.819,
//     positive: 0.181,
//     protein: 23.0,
//     rating: 5,
//     recipe_id: 10545,
//     review:
//       "i have been looking for a good pumpkin pie recipe for three years and stumbled across this recipe for thanksgiving. my husband said i hit the mark with this recipe. the perfect blend of spices with good texture. i followed the recipe as it and added a bit of additional cinnamon.",
//     saturated_fat: 40.0,
//     sodium: 27.0,
//     steps:
//       "combine pumpkin and remaining ingredients in a large bowl\nbeat at medium speed with an electric mixer 2 minutes\npour into prepared piecrust\nbake at 425 degrees for 15 minutes\nreduce heat to 350 degrees\nbake 50 additional minutes or until a knife inserted in center comes out clean\ncool on a wire rack",
//     submitted: "2002-05-18",
//     sugar: 316.0,
//     tags: "weeknight, time-to-make, course, preparation, occasion, healthy, pies-and-tarts, desserts, oven, heirloom-historical, holiday-event, pies, dietary, thanksgiving, equipment, number-of-servings, 4-hours-or-less",
//     total_fat: 30.0,
//     user_id: 38690,
//   },
// ];

function App(): JSX.Element {
  const [data, setData] = React.useState<RecipeData[]>([]);
  const [loading, setLoading] = React.useState<boolean>(false);
  const [model, setModel] = React.useState<string>("hybrid");
  const [recipeSearch, setRecipeSearch] = React.useState<string>("ice cream");
  const [suggestions, setSuggestions] = React.useState<Partial<RecipeData[]>>(
    []
  );
  const [recipeId, setRecipeId] = React.useState<number>(-1);

  useEffect(() => {
    const fetchData = async (): Promise<void> => {
      setLoading(true);
      console.log("Fetching data...");
      const response = await fetch(
        `http://127.0.0.1:5000/model/${model}?recipe_id=${recipeId}`
      );
      const data = await response.json();
      setData(data);
      setLoading(false);
    };
    console.log(recipeId, model);
    if (recipeId !== -1 && model.length > 0) {
      void fetchData();
    }
  }, [recipeId, model]);

  const search = async (): Promise<void> => {
    console.log("Searching...");
    const fetchSuggestions = async (): Promise<void> => {
      const response = await fetch(
        `http://127.0.0.1:5000/search?query=${recipeSearch}`
      );
      const data = await response.json();
      const names = data.map((recipe: RecipeData) => ({
        name: recipe.name,
        recipe_id: recipe.recipe_id,
      }));
      if (!(names.length <= 10 && names[0].name === recipeSearch)) {
        setSuggestions(names);
      }
    };
    if (recipeSearch.length > 0) {
      void fetchSuggestions();
    }
  };

  useDebounce(
    () => {
      void search();
    },
    [recipeSearch],
    500
  );

  const handleSuggestionClick = (suggestion: Partial<RecipeData>): void => {
    console.log("Suggestion clicked");
    setRecipeSearch((suggestion as RecipeData).name.toString());
    setRecipeId((suggestion as RecipeData).recipe_id);
    setSuggestions([]);
  };

  return (
    <>
      <div className="App">
        <h1 className="text-2xl text-center p-8 uppercase">
          Food Recommendation
        </h1>
        <div className="flex flex-row w-full">
          <div className="mx-auto relative">
            <input
              type="text"
              placeholder="Search using id e.g (192839)"
              className="border-2 border-gray-300 rounded-md p-2 m-2"
              onChange={(e) => setRecipeSearch(e.target.value)}
              value={recipeSearch}
            />

            {suggestions.length > 0 && (
              <ul className="absolute top-full left-0 w-full mt-1 bg-white rounded-md shadow-md z-10">
                {suggestions.map((suggestion) => (
                  <li
                    key={suggestion?.name}
                    className="px-4 py-2 cursor-pointer hover:bg-gray-100"
                    onClick={() =>
                      handleSuggestionClick(suggestion as RecipeData)
                    }
                  >
                    {suggestion?.name}
                  </li>
                ))}
              </ul>
            )}
            <button className="border-2 border-gray-300 rounded-md p-2 m-2">
              Search
            </button>
            <button className="border-2 border-gray-300 rounded-md p-2 m-2">
              Clear
            </button>
          </div>
        </div>
        {/*    Dropdown */}
        <div className="flex flex-row w-full">
          <div className="mx-auto">
            <select
              className="border-2 border-gray-300 rounded-md p-2 m-2"
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="-1" disabled selected>
                Recommendation Model
              </option>
              <option value="hybrid">Hybrid</option>
              <option value="nmf">NMF</option>
              <option value="cdl">CDL</option>
            </select>
          </div>
        </div>
      </div>
      <Recipes recipes={data} />
    </>
  );
}

export default App;
