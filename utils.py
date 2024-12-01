import os
os.environ['HF_HOME'] = "/scratch/ssd004/scratch/lfy"

from llama_cpp import LlamaGrammar
import pyterrier as pt
import random, os
import numpy as np
import pandas as pd
import torch
import json
import re
from rapidfuzz import process
import tempfile
from tqdm import tqdm

from whoosh import index, writing
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import *
from whoosh.qparser import QueryParser

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
    df_nutritions = df_nutritions.rename(columns={'irom': 'iron', 'zink': 'zinc'})
    num_nutrition_with_missing_info = df_nutritions.isnull().any(axis=1).sum()
    print(f"Number of recipes with missing info: {num_nutrition_with_missing_info}")
    missing_per_column_nutrition = df_nutritions.isnull().sum()
    missing_per_column_nutrition = missing_per_column_nutrition[missing_per_column_nutrition > 0]
    print(f"Missing values per column:\n{missing_per_column_nutrition}")
    df_nutritions.fillna(0, inplace=True)
    return df_nutritions

def nutrients_dict_init():
    return {'calories': 0, 'total_fat': 0, 'fat': 0, 'saturated_fat': 0, 'saturated_fatty_acids': 0, 'cholesterol': 0,'sodium': 0, 'carbohydrate': 0, 'fiber': 0, 'sugars': 0, 'protein': 0, 'vitamin_a': 0, 'vitamin_a_rae': 0,'carotene_alpha': 0, 'carotene_beta': 0, 'cryptoxanthin_beta': 0, 'lutein_zeaxanthin': 0, 'lucopene': 0,'vitamin_b12': 0, 'vitamin_b6': 0, 'vitamin_c': 0, 'vitamin_d': 0, 'vitamin_e': 0, 'tocopherol_alpha': 0,'vitamin_k': 0, 'calcium': 0, 'copper': 0, 'iron': 0, 'magnesium': 0, 'manganese': 0, 'phosphorous': 0,'potassium': 0, 'selenium': 0, 'zinc': 0, 'water': 0}

def nutrients_to_str(recipe_with_nutrients, nutrients_dict):
    for ing in recipe_with_nutrients:
        for nu in nutrients:
            nutrients_dict[nu] += ing[nu]

    nutrients_str = ""
    for nu in nutrients:
        nutrients_str += nu
        nutrients_str += ": "
        nutrients_str += str(round(nutrients_dict[nu], 1))
        nutrients_str += " "
        nutrients_str += nutrients_unit[nu]
        nutrients_str += ", "
    return nutrients_str
    
def get_query_res(result_json):

    recipes_full = pd.read_csv('data/recipes.csv', encoding='utf-8', on_bad_lines='skip')
    recipes_full = recipes_full.fillna(value={'RecipeServings': 1.0})

    recipe_cols = ["Name", "Description", "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"]
    recipes_full["full_document"] = recipes_full[recipe_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    recipes_full = recipes_full.astype({'RecipeId': 'string'})

    docs_df = recipes_full[['RecipeId', 'full_document']].copy()
    docs_df = docs_df.rename(columns={"RecipeId": "docno", "full_document": "text"})


    mySchema = Schema(index = ID(stored=True),
                    content = TEXT(stored=True))
    
    indexDir = tempfile.mkdtemp()

    ix = index.create_in(indexDir, mySchema)

    writer = ix.writer()

    for i in tqdm(range(len(docs_df))):
        writer.add_document(index=str(docs_df.docno.iloc[i]), content=str(docs_df.text.iloc[i]))
    writer.commit()

    query_results = []

    for food in tqdm(result_json['food_names']):
        parser = QueryParser("content", schema=ix.schema)
        searcher = ix.searcher()
        query = parser.parse(food)
        results = searcher.search(query, limit=1)

        for res in results:
            res_dict = {}
            res_dict['index'] = res['index']
            res_dict['query'] = food
            res_dict['Name'] = recipes_full[recipes_full['RecipeId'] == res['index']]['Name'].to_string(index=False)

            res_dict['RecipeInstructions'] = res['content']

            query_results.append(res_dict)

    return query_results

ask_food_prompt_template = "The user is living in {user_location}. The user is looking for a {user_query}. {extra_input} Suggest 5 {user_query} which can be cooked by the user, without actual recipe."

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

ask_recipe_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}. {extra_input} \nI will provide a few recipes below.\nGenerate a new recipe that satisfy user nutrition requirements.\nYou can reuse the existing recipe, or you can modify it or create a new recipe.\nHere are few potential recipes: {retrieved_recipes}."

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

ask_ingreds_g_template = "The user is living in {user_location}. The user is looking for a {user_query}. {extra_input} Here is a generated recipe: {generated_recipes}. Generate all ingredients with their corresponding unit and amount in the recipe. For each ingredient, list their name, and their weight in grams."

ask_ingreds_template = "The user is living in {user_location}. The user is looking for a {user_query}. {extra_input} Here is a generated recipe: {generated_recipes}. Generate all ingredients with their corresponding unit and amount in the recipe. For each ingredient, list their name, the appropriate unit to measure, and their amount."

schema_ingreds_g = r'''
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
ingredset ::= ("{" ingredname "," space ingredunit "," space ingredamt space "}")
ingreds ::= (space "[" space (ingredset ("," space ingredset){20})? "]")
'''

enforce_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}. {extra_input}\nHere is a generated recipe: {generated_recipes}.\nHere is the nutrition information of the recipe: {nutrients_str}. Given the nutrition information, does it comply with the user requirements? Your will return true or false in the field of Compliance and return the reason in the Reason field."

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

fix_recipe_template = "The user is living in {user_location}.\nThe user is looking for a {user_query}. {extra_input} \nHere is a generated recipe: {generated_recipes}.\nHere is the nutrition information of the recipe: {nutrients_str}. It is not complying with the nutrition or dietary requirements because {comply_reason}. Fix the recipe and generate a new recipe. You must not include nutrition information or any notes. Stop generating instructions when the food is ready to serve. Do not put notes or nutrition in cooking instructions. Your output will be in the format:"


nutrients = ['calories', 'total_fat', 'fat', 'saturated_fat', 'saturated_fatty_acids', 'cholesterol','sodium', 'carbohydrate', 'fiber', 'sugars', 'protein', 'vitamin_a', 'vitamin_a_rae','carotene_alpha', 'carotene_beta', 'cryptoxanthin_beta', 'lutein_zeaxanthin', 'lucopene','vitamin_b12', 'vitamin_b6', 'vitamin_c', 'vitamin_d', 'vitamin_e', 'tocopherol_alpha','vitamin_k', 'calcium', 'copper', 'iron', 'magnesium', 'manganese', 'phosphorous','potassium', 'selenium', 'zinc', 'water']

nutrients_unit = {'calories': '',
                  'total_fat': 'g',
                  'fat': 'g',
                  'saturated_fat': 'g',
                  'saturated_fatty_acids': 'g',
                  'cholesterol': '',
                  'sodium': 'mg',
                  'carbohydrate': 'g',
                  'fiber': 'g',
                  'sugars': 'g',
                  'protein': 'g',
                  'vitamin_a': 'IU',
                  'vitamin_a_rae': 'mcg',
                  'carotene_alpha': 'mcg',
                  'carotene_beta': 'mcg',
                  'cryptoxanthin_beta': 'mcg',
                  'lutein_zeaxanthin': 'mcg',
                  'lucopene': '',
                  'vitamin_b12': 'mcg',
                  'vitamin_b6': 'mg',
                  'vitamin_c': 'mg',
                  'vitamin_d': 'IU',
                  'vitamin_e': 'mg',
                  'tocopherol_alpha': 'mg',
                  'vitamin_k': 'mcg',
                  'calcium': 'mg',
                  'copper': 'mg',
                  'iron': 'mg',
                  'magnesium': 'mg',
                  'manganese': 'mg',
                  'phosphorous': 'mg',
                  'potassium': 'mg',
                  'selenium': 'mcg',
                  'zinc': 'mg',
                  'water': 'g'}

