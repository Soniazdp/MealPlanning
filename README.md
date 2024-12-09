# MealPlanning

This repository provides a tool for generating customized recipes that follows the nutrition requirements by combining IR and LLM agents.

### Dependencies:
```
llama_cpp_python
numpy
pandas
torch
rapidfuzz
whoosh
transformers
```

### Get started
```
main.py
```
`main.py` provides a minimum example for using the library. Necessary input includes

- `user-location` for the geographical location of the user.
- `user-query` for basic input, i.e. Vegan Korean Food.
- `ingreds-path` for ingredient csv file output.
- `instructs-path` for cooking steps and nutrition information file output.

Extra user input can be passed into `extra-input` as a string. More details can be accessed through `python main.py -h`. Example outputs are provided as `cook_steps.json` and `ingredients_result.csv`.
