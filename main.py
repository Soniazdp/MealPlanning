import os
os.environ['HF_HOME'] = "/scratch/ssd004/scratch/lfy"

from llama_cpp import Llama
import json
import pyterrier as pt
import csv
import re
import numpy as np
import pandas as pd
import utils


utils.seed_everything(42)

llm = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    # n_gpu_layers=-1,
    n_ctx=128000,
    verbose=False,
    cache_dir="/checkpoint/lfy/14024697"
)

temperature = 0.7
user_location = "California"
user_query = "vegan korean food"
extra_input = ""

ask_food_prompt = utils.ask_food_prompt_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input)

result_json = utils.generate(llm, ask_food_prompt, utils.schema_food_name, temperature)

print("Food names: ")
print(result_json)

extracted_recipe = {}

query_res = utils.get_query_res()

for recipe in query_res:
    recipe_name = recipe['Name']
    recipe_instructions = recipe["RecipeInstructions"]
    matches = re.findall(r'"(.*?)"', recipe_instructions)
    recipe_instructions = tuple(matches)
    steps_dict = {f"Step {i+1}": step for i, step in enumerate(recipe_instructions)}
    steps_json = json.dumps(steps_dict, indent=2)
    extracted_recipe[recipe_name] = {} # Overwrite recipe with same name, no duplicate
    extracted_recipe[recipe_name]['instructions'] = steps_json

ask_recipe_prompt = utils.ask_recipe_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, retrieved_recipes=extracted_recipe)

cook_steps = utils.generate_recipe(llm, ask_recipe_prompt, utils.schema_recipe, temperature)

print("Recipe steps: ")
print(cook_steps)

concat_steps = utils.steps_to_str(cook_steps)

ask_ingreds_g_prompt = utils.ask_ingreds_g_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

ask_ingreds_prompt = utils.ask_ingreds_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

ingreds = utils.generate(llm, ask_ingreds_g_prompt, utils.schema_ingreds_g, temperature)

res_ingreds = utils.generate(llm, ask_ingreds_prompt, utils.schema_ingreds, temperature)

print("Ingredients: ")
print(ingreds)

ingreds['Ingredients']

nutrients = utils.nutrients

df_nutritions = utils.get_df_nutritions("data/nutrition.csv")

recipe_with_nutrients = utils.get_best_match(ingreds['Ingredients'], df_nutritions, nutrients)

nutrients_dict = utils.nutrients_dict_init()

nutrients_str = utils.nutrients_to_str(recipe_with_nutrients, nutrients_dict)

enforce_prompt = utils.enforce_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str)

compliance_res = utils.generate(llm, enforce_prompt, utils.schema_enforce, temperature)

print("Compliance: ")
print(compliance_res)

count = 0

while compliance_res['Compliance'] == False and count < 10:

    print("Trial: ", count)

    fix_recipe_prompt = utils.fix_recipe_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str, comply_reason=compliance_res['Reason'])

    cook_steps = utils.generate_recipe(llm, fix_recipe_prompt, utils.schema_recipe, temperature)

    print("Recipe steps: ")
    print(cook_steps)

    concat_steps = utils.steps_to_str(cook_steps)

    ask_ingreds_g_prompt = utils.ask_ingreds_g_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

    ask_ingreds_prompt = utils.ask_ingreds_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

    ingreds = utils.generate(llm, ask_ingreds_g_prompt, utils.schema_ingreds_g, temperature)

    res_ingreds = utils.generate(llm, ask_ingreds_prompt, utils.schema_ingreds, temperature)

    print("Ingredients: ")
    print(ingreds)

    recipe_with_nutrients = utils.get_best_match(ingreds['Ingredients'], df_nutritions, nutrients)

    nutrients_dict = utils.nutrients_dict_init()

    nutrients_str = utils.nutrients_to_str(recipe_with_nutrients, nutrients_dict)

    enforce_prompt = utils.enforce_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str)

    compliance_res = utils.generate(llm, enforce_prompt, utils.schema_enforce, temperature)

    print("Compliance: ")
    print(compliance_res)

    count += 1

cook_steps['Nutrition info'] = recipe_with_nutrients
ingreds_df = pd.DataFrame(res_ingreds['Ingredients'])
ingreds_df.to_csv("ingredients_result.csv", index=False)
json_object = json.dumps(cook_steps, indent=4)
with open("cook_steps.json", "w") as outfile:
    outfile.write(json_object)