import json
from llm import run_recipe_generation
from RAG_test import analyze_ingredients_matching

def main():
    # 질문할 때마다 recipes.json, ingredients.json 초기화
    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False)
    with open("ingredients.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False)
    query = input("검색할 레시피 또는 질문을 입력하세요: ")
    # 1. LLM으로 레시피 생성 및 재료 추출
    recipe_info = run_recipe_generation(query)
    if not recipe_info:
        print("레시피 생성 실패")
        return

    # 2. 파일 로드
    with open("materials.json", "r", encoding="utf-8") as f:
        materials = json.load(f)
    with open("recipes.json", "r", encoding="utf-8") as f:
        recipe_data = json.load(f)
    with open("ingredients.json", "r", encoding="utf-8") as f:
        ingredients_data = json.load(f)

    # 3. 재료 매칭 분석 (RAG_test.py 함수 활용)
    match_result = analyze_ingredients_matching(materials, recipe_data, ingredients_data)

    # 4. 결과 종합 및 출력
    print("\n=== 레시피 ===")
    print(recipe_info["recipe"])
    print("\n=== 재료 매칭 결과 ===")
    print("정확히 일치:")
    for item in match_result["exact_matches"]:
        print(f"- {item['ingredient']}: {item['location']}")
    print("유사:")
    for item in match_result["similar_matches"]:
        print(f"- {item['ingredient']} → {item['mart_item']} ({item['location']})")
    print("구할 수 없음:")
    for item in match_result["unavailable"]:
        print(f"- {item}")

if __name__ == "__main__":
    main()