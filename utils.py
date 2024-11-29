import os
os.environ['HF_HOME'] = "/scratch/ssd004/scratch/lfy"

from llama_cpp import LlamaGrammar
import random, os
import numpy as np
import pandas as pd
import torch
import json
import re
from rapidfuzz import process

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate(llm, prompt, schema, temperature):
    grammar = LlamaGrammar.from_string(grammar=schema, verbose=False)
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        grammar=grammar,
        temperature=temperature,
    )
    result_json = json.loads(result['choices'][0]['message']['content'])
    return result_json

def generate_recipe(llm, prompt, schema, temperature):
    grammar = LlamaGrammar.from_string(grammar=schema, verbose=False)
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON. You must not include nutrition information or any notes. Stop generating instructions when the food is ready to serve. Do not put notes or nutrition in cooking instructions. Your output will be in the format: {\"Description\": A brief description of the recipe, \"Cooking Instructions\": {\"Step 1\": first cooking step, \"Step 2\": second cooking step, ...}}",
            },
            {"role": "user", "content": prompt},
        ],
        grammar=grammar,
        temperature=temperature,
    )
    result_json = json.loads(result['choices'][0]['message']['content'])
    return result_json

def get_best_match(ingredients, df, important_nutrients, threshold=80):
    # choices = df['name'].tolist()
    name_index = df.set_index('name').to_dict(orient='index')

    result = []
    for ingredient in ingredients:
        best_match, score, _ = process.extractOne(ingredient['Ingredient name'], name_index.keys())

        # if the score doesn't reach threshold, there is no valid match found for ingredient
        if score < threshold:
            print(f"No valid match found for {ingredient['Ingredient name']}")
            # put 0 for all important nutrients
            for important_nutrient in important_nutrients:
                ingredient[important_nutrient] = 0
            result.append(ingredient)
            continue

        # combine the columns in df['name'] == best_match to key and value pairs in the dictionary
        # only add the key and value pairs where key exists in important_nutrients
        # best_match_row = df[df['name'] == best_match].to_dict('records')[0]
        best_match_row = name_index[best_match]
        for key, value in best_match_row.items():

            if key in important_nutrients:
                # print(key, value)
                # print(isinstance(value, str))

                if isinstance(value, str):  # If the value is a string, check for units
                    # print(key, value)
                    # Use regular expression to remove units like 'g', 'mg', 'IU', etc.
                    if key != 'calories':
                        value = re.sub(r'[^\d.-]', '', value)  # Remove anything that's not a digit or decimal point
                    # add value to ingredient
                    ingredient[key] = float(value) * float(ingredient['Grams']) / 100 if value else 0.0
                else:
                    # print(key, value)
                    ingredient[key] = value * float(ingredient['Grams']) / 100 if value else 0.0

        result.append(ingredient)

    return result

def steps_to_str(steps):
    concat_steps = steps['Description']
    concat_steps += '\n'

    for step in steps['Cooking Instructions']:
        # concat_steps.append()
        concat_steps += step
        concat_steps += ': '
        concat_steps += steps['Cooking Instructions'][step]
        concat_steps += '\n'
    concat_steps = concat_steps[:-1]
    return concat_steps

def get_df_nutritions(path):
    df_nutritions = pd.read_csv(path)
    num_nutrition_with_missing_info = df_nutritions.isnull().any(axis=1).sum()
    print(f"Number of recipes with missing info: {num_nutrition_with_missing_info}")
    missing_per_column_nutrition = df_nutritions.isnull().sum()
    missing_per_column_nutrition = missing_per_column_nutrition[missing_per_column_nutrition > 0]
    print(f"Missing values per column:\n{missing_per_column_nutrition}")
    df_nutritions.fillna(0, inplace=True)
    return df_nutritions

def nutrients_dict_init():
    return {'calories': 0, 'total_fat': 0, 'fat': 0, 'saturated_fat': 0, 'saturated_fatty_acids': 0, 'cholesterol': 0,'sodium': 0, 'carbohydrate': 0, 'fiber': 0, 'sugars': 0, 'protein': 0, 'vitamin_a': 0, 'vitamin_a_rae': 0,'carotene_alpha': 0, 'carotene_beta': 0, 'cryptoxanthin_beta': 0, 'lutein_zeaxanthin': 0, 'lucopene': 0,'vitamin_b12': 0, 'vitamin_b6': 0, 'vitamin_c': 0, 'vitamin_d': 0, 'vitamin_e': 0, 'tocopherol_alpha': 0,'vitamin_k': 0, 'calcium': 0, 'copper': 0, 'magnesium': 0, 'manganese': 0, 'phosphorous': 0,'potassium': 0, 'selenium': 0, 'water': 0}

def nutrients_to_str(recipe_with_nutrients, nutrients_dict):
    for ing in recipe_with_nutrients:
        for nu in nutrients:
            nutrients_dict[nu] += ing[nu]

    nutrients_str = ""
    for nu in nutrients:
        nutrients_str += nu
        nutrients_str += ": "
        nutrients_str += str(round(nutrients_dict[nu], 1))
        nutrients_str += ", "
    return nutrients_str
    
ask_food_prompt_template = "The user is living in {user_location}. The user is looking for a {user_query}. Suggest 5 {user_query} which can be cooked by the user, without actual recipe."

schema_food_name = r'''
root ::= (
    "{" newline
        doublespace "\"food_names\":" space listofstring newline
    "}"
)
newline ::= "\n"
doublespace ::= "  "
number ::= [0-9]+   "."?   [0-9]*
boolean ::= "true" | "false"
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
listofstring ::= ("[" space (string ("," space string){4})? "]")
'''

ask_recipe_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}.\nI will provide a few recipes below.\nGenerate a new recipe that satisfy user nutrition requirements.\nYou can reuse the existing recipe, or you can modify it or create a new recipe.\nHere are few potential recipes: {retrieved_recipes}."

schema_recipe = r'''
root ::= (
    "{" newline
        doublespace "\"Description\":" space string "," newline
        doublespace "\"Cooking Instructions\":" cookinstructs space 
    "}"
)
newline ::= "\n"
doublespace ::= "  "
number ::= [0-9]+   "."?   [0-9]*
integer ::= [0-9]*
boolean ::= "true" | "false"
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
sentence ::= char* space
listofstring ::= ("[" space (string ("," space string)*)? "]")
cookstep ::= ("\"Step" space integer "\"" ":" space string)
cookinstructs ::= (space "{" space (cookstep ("," space cookstep){10})? "}")
'''

ask_ingreds_template = "The user is living in {user_location}. The user is looking for a {user_query}. Here is a generated recipe: {generated_recipes}. Generate all ingredients with their corresponding unit and amount in the recipe. For each ingredient, list their name, and their weight in grams."


schema_ingreds = r'''
root ::= (
    "{" newline
        doublespace "\"Ingredients\":" space ingreds newline
    "}"
)
newline ::= "\n"
doublespace ::= "  "
number ::= [0-9]+   "."?   [0-9]*
integer ::= [0-9]*
boolean ::= "true" | "false"
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
sentence ::= char* space
listofstring ::= ("[" space (string ("," space string)*)? "]")
cookstep ::= ("\"Step" space integer "\"" ":" space string)
cookinstructs ::= (space "{" space (cookstep ("," space cookstep){10})? "}")
ingredname ::= ("\"Ingredient name" space "\"" ":" space string)
ingredunit ::= ("\"Unit" space "\"" ":" space string)
ingredamt ::= ("\"Amount" space "\"" ":" space number)
ingredwgt ::= ("\"Grams" space "\"" ":" space number)
ingredset ::= ("{" ingredname "," space ingredwgt space "}")
ingreds ::= (space "[" space (ingredset ("," space ingredset){20})? "]")
'''

enforce_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}.\nHere is a generated recipe: {generated_recipes}.\nHere is the nutrition information of the recipe: {nutrients_str}. Given the nutrition information, does it comply with the user requirements? Your will return true or false in the field of Compliance and return the reason in the Reason field."

schema_enforce = r'''
root ::= (
    "{" newline
        doublespace "\"Compliance\":" space boolean "," newline
        doublespace "\"Reason\":" space string newline
    "}"
)
newline ::= "\n"
doublespace ::= "  "
number ::= [0-9]+   "."?   [0-9]*
integer ::= [0-9]*
boolean ::= "true" | "false"
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
sentence ::= char* space
listofstring ::= ("[" space (string ("," space string)*)? "]")
cookstep ::= ("\"Step" space integer "\"" ":" space string)
cookinstructs ::= (space "{" space (cookstep ("," space cookstep){10})? "}")
ingredname ::= ("\"Ingredient name" space "\"" ":" space string)
ingredunit ::= ("\"Unit" space "\"" ":" space string)
ingredamt ::= ("\"Amount" space "\"" ":" space number)
ingredwgt ::= ("\"Grams" space "\"" ":" space number)
ingredset ::= ("{" ingredname "," space ingredwgt space "}")
ingreds ::= (space "[" space (ingredset ("," space ingredset){20})? "]")
'''

fix_recipe_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}.\nHere is a generated recipe: {generated_recipes}.\nHere is the nutrition information of the recipe: {nutrients_str}. It is not complying with the nutrition or dietary requirements because {comply_reason}. Fix the recipe and generate a new recipe. You must not include nutrition information or any notes. Stop generating instructions when the food is ready to serve. Do not put notes or nutrition in cooking instructions. Your output will be in the format:"


nutrients = ['calories', 'total_fat', 'fat', 'saturated_fat', 'saturated_fatty_acids', 'cholesterol','sodium', 'carbohydrate', 'fiber', 'sugars', 'protein', 'vitamin_a', 'vitamin_a_rae','carotene_alpha', 'carotene_beta', 'cryptoxanthin_beta', 'lutein_zeaxanthin', 'lucopene','vitamin_b12', 'vitamin_b6', 'vitamin_c', 'vitamin_d', 'vitamin_e', 'tocopherol_alpha','vitamin_k', 'calcium', 'copper', 'magnesium', 'manganese', 'phosphorous','potassium', 'selenium', 'water']


'''
recipes_full = pd.read_csv('data/recipes.csv', encoding='utf-8', on_bad_lines='skip')
recipes_full = recipes_full.fillna(value={'RecipeServings': 1.0})

recipes_full.head()

recipe_cols = ["Name", "Description", "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"]
recipes_full["full_document"] = recipes_full[recipe_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
recipes_full = recipes_full.astype({'RecipeId': 'string'})

docno = len(recipes_full)

docs_df = recipes_full[['RecipeId', 'full_document']].copy()
docs_df = docs_df.rename(columns={"RecipeId": "docno", "full_document": "text"})
docs_df.head()

index_dir = './recipes_index'
indexer = pt.DFIndexer(index_dir, overwrite=True, )
index_ref = indexer.index(docs_df["text"], docs_df["docno"])
index_ref.toString()
index = pt.IndexFactory.of(index_ref)
br = pt.BatchRetrieve(index, wmodel="Tf")


query_set = result_json['food_names']

desc = ['vegan', 'Vegan', 'Vegetarian', 'vegetarian']

for d in desc:
  query_set = [s.strip(d) for s in query_set]

breakpoint()

query_results = br.transform(query_set)

query_results.head()

result_cols = ["RecipeId", "Name", "Description", "RecipeIngredientQuantities", "RecipeIngredientParts",
               "RecipeInstructions", "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
               "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent", "RecipeServings"]

breakpoint()

results = pd.merge(query_results[["docno", "rank", "query"]],
                         recipes_full[result_cols],
                  right_on='RecipeId', left_on='docno', how='left')

results[results["query"] == query_set[0]].head()[['query', 'Name', 'RecipeInstructions']]

query_res = []

for query in query_set:
  res = results[results["query"] == query].head()[['query', 'Name', 'RecipeInstructions']]
  query_res.extend(res.to_dict('records'))

print(query_res)
'''

query_res = [{"index":0,"query":"kimichi stew","Name":"Perfect Chicken Stew","RecipeInstructions":"c(\"Season a 3-7 pound chicken with Garlic powder and Pepper. Roast chicken in oven at 325 degrees.\", \"While chicken is cooking, dice potatoes, slice carrots, chop onions and carrots to desired thickness. Place vegetables in stewing pot and add water until vegetables are covered with about an 3 inches of water. Boil rapidly until potatoes are just finished.\", \"Remove vegetables from the pot by straining them and keep the water. By removing the vegetables and letting them cool, you prevent overcooking them and they won't dissolve into nothing.\", \n\"With remaining water on low heat, add can of cream of mushroom soup, can of chicken stock and milk (milk optional, Zie Ga Zink).\", \"If you don't use milk, I suggest a premium ready to serve brand of creamed mushroom soup, it will be of a smoother, creamier consistency than the regular cans of mushroom soup.\", \"Get a small sealable container and fill with 1 cup of cold water, then add 1 cup of flour, cover and seal, then immediately shake vigorously. You are making a thickener for the stew, it should look like the consistency of glue with no lumps. If to thick add a bit of water, too thin add a bit more flour, shake very hard again. If there are a few lumps you can remove them by straining. This process, once learned, is very useful for making gravies or other stews without using a high-fat butter and flour 'roux' thickener.\", \n\"Rapidly add thickener to the starch water/mushroom soup/stock/milk mixture using a whisk. You may have to make a little more thickener if you want a hardier stew, just remember that the stew will thicken more after it is removed from the heat and it stands. Simmer to desired consistency. Stir often. Do not burn! I suggest a non-stick stew pot, it helps prevent burning.\", \"Add the cooked (now cooled) vegetables to the stew.\", \"When chicken is finished roasting, drain juices into the stew. Remove skin and bones.  Tear or cut chicken apart and add to the stew.\", \n\"Stir in about 2-3 tablespoons of salt to stew  and about the same amount of pepper to taste.\", \"If you want, try adding a dash of hot sauce or a pinch of Sambel Olek.\", \"Let stew simmer for a little longer. Serve with fresh bread and Enjoy.\", \"Questions? brennarlauterbach@hotmail.com.\")"},{"index":1,"query":"kimichi stew","Name":"Wintry Beef Vegetable Stew With Fluffy Herb Dumplings","RecipeInstructions":"c(\"Cook and stir beef in shortening in heavy 8-10 quart stock pot, until beef is well browned. (Note: If too much liquid builds up to prevent adequate browning, pour off excess liquid into a bowl and reserve. Continue to brown the beef and when well browned, add the reserved liquid back into the pot.).\", \"Add 5 cups hot water, 1/2 teaspoon salt and the black pepper.\", \"Heat to boiling; reduce heat.\", \"Cover and simmer until beef is almost tender, 45 minutes to 1 hour.\", \"Stir in potato, turnip, rutabaga, carrots, green pepper, green beans (if using), celery, onion, bouquet sauce, the bouillon cube and bay leaves.\", \n\"Cover and simmer until vegetables are tender (but do not overcook), stirring once, about 25 minutes.\", \"Prepare dough (see below) for Dumplings;  set aside.\", \"Using a fork, blend together 1 cup cold water and the 4 tablespoons flour in a small mixing bowl; stir gradually into stew.\", \"Heat to boiling, stirring constantly.\", \"Boil and stir 1 minute; reduce heat.\", \"Do ahead tip: After boiling and stirring 1 minute, stew can be covered and refrigerated no longer than 48 hours. To serve, heat to boiling over medium-high heat. Continue as directed.\", \n\"DUMPLINGS:\", \"In a large bowl, cut shortening into combined flour, baking powder, salt, parsley and herbs until mixture resembles fine crumbs.\", \"Stir in milk.\", \"Drop by heaping tablespoons onto hot meat or vegetables in boiling stew (do not drop directly into liquid).\", \"Cook uncovered 15 minutes.\", \"Cover and cook about 15 minutes longer. Cut a dumpling in half to test for doneness; you want them done but not dry!\", \"Serve stew piping hot, with a buttered baguette and a glass of cider, ale, or wine. As with all good stews, this stew is even better reheated the next day, after flavors have had a chance to meld. Stew leftovers freeze and reheat beautifully, and would make a delicious cottage or shepherd's pie.\"\n)"}]