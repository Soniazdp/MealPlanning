from llama_cpp import Llama
import json
import csv
import re
import numpy as np
import pandas as pd
import utils
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default=0.7, help="Temperature for model generation.")
    parser.add_argument("--user-location", default="California", help="User geographical location.")
    parser.add_argument("--user-query", default="Vegan Korean Food", help="User input query.")
    parser.add_argument("--extra-input", default="", help="Extra user input.")
    parser.add_argument("--seed", default=42, help="Random seed.")
    parser.add_argument("--model", default="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF", help="Model name or path.")
    parser.add_argument("--filename", default="*Q4_K_M.gguf", help="Specific quantization filename.")
    parser.add_argument("--ingreds-path", default="ingreds.csv", help="Output file path for ingredients (csv).")
    parser.add_argument("--instructs-path", default="instructions.json", help="Output file path for cooking instructions and nutrition information (json).")
    parser.add_argument("--verbose", default=False, help="Print intermediate outputs.")
    return parser.parse_args()

def main(args):
    utils.seed_everything(args.seed)

    llm = Llama.from_pretrained(
        repo_id=args.model,
        filename=args.filename,
        n_gpu_layers=-1,
        n_ctx=128000,
        verbose=args.verbose
    )

    temperature = args.temperature
    user_location = args.user_location
    user_query = args.user_query
    extra_input = args.extra_input

    ask_food_prompt = utils.ask_food_prompt_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input)

    result_json = utils.generate(llm, ask_food_prompt, utils.schema_food_name, temperature)
    
    if args.verbose:
        print("Food names: ")
        print(result_json)

    extracted_recipe = {}

    query_res = utils.get_query_res(result_json)

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

    if args.verbose:
        print("Recipe steps: ")
        print(cook_steps)

    concat_steps = utils.steps_to_str(cook_steps)

    ask_ingreds_g_prompt = utils.ask_ingreds_g_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

    ask_ingreds_prompt = utils.ask_ingreds_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

    ingreds = utils.generate(llm, ask_ingreds_g_prompt, utils.schema_ingreds_g, temperature)

    res_ingreds = utils.generate(llm, ask_ingreds_prompt, utils.schema_ingreds, temperature)

    if args.verbose:
        print("Ingredients: ")
        print(ingreds)

    ingreds['Ingredients']

    nutrients = utils.nutrients

    df_nutritions = utils.get_df_nutritions("data/nutrition.csv")

    recipe_with_nutrients = utils.get_best_match(ingreds['Ingredients'], df_nutritions, nutrients)

    nutrients_dict = utils.nutrients_dict_init()

    nutrients_str, nutrients_dict = utils.nutrients_to_str(recipe_with_nutrients, nutrients_dict)

    enforce_prompt = utils.enforce_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str)

    compliance_res = utils.generate(llm, enforce_prompt, utils.schema_enforce, temperature)

    if args.verbose:
        print("Compliance: ")
        print(compliance_res)

    count = 0

    while compliance_res['Compliance'] == False and count < 10:

        if args.verbose: print("Trial: ", count)

        fix_recipe_prompt = utils.fix_recipe_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str, comply_reason=compliance_res['Reason'])

        cook_steps = utils.generate_recipe(llm, fix_recipe_prompt, utils.schema_recipe, temperature)

        if args.verbose:
            print("Recipe steps: ")
            print(cook_steps)

        concat_steps = utils.steps_to_str(cook_steps)

        ask_ingreds_g_prompt = utils.ask_ingreds_g_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

        ask_ingreds_prompt = utils.ask_ingreds_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps)

        ingreds = utils.generate(llm, ask_ingreds_g_prompt, utils.schema_ingreds_g, temperature)

        res_ingreds = utils.generate(llm, ask_ingreds_prompt, utils.schema_ingreds, temperature)

        if args.verbose:
            print("Ingredients: ")
            print(ingreds)

        recipe_with_nutrients = utils.get_best_match(ingreds['Ingredients'], df_nutritions, nutrients)

        nutrients_dict = utils.nutrients_dict_init()

        nutrients_str, nutrients_dict = utils.nutrients_to_str(recipe_with_nutrients, nutrients_dict)

        enforce_prompt = utils.enforce_template.format(user_location=user_location, user_query=user_query, extra_input=extra_input, generated_recipes=concat_steps, nutrients_str=nutrients_str)

        compliance_res = utils.generate(llm, enforce_prompt, utils.schema_enforce, temperature)
        
        if args.verbose:
            print("Compliance: ")
            print(compliance_res)

        count += 1

    cook_steps['Nutrition sum'] = nutrients_dict
    cook_steps['Nutrition info'] = recipe_with_nutrients
    ingreds_df = pd.DataFrame(res_ingreds['Ingredients'])
    ingreds_df.to_csv(args.ingreds_path, index=False)
    json_object = json.dumps(cook_steps, indent=4)
    with open(args.instructs_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    args = parse_args()
    main(args)
