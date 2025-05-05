from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json
import os

from location import extract_materials_from_rag_result
from RAG_test import analyze_ingredients_matching

# --- 파일 경로 설정 ---
RECIPE_FILE = "recipes.json"
INGREDIENTS_FILE = "ingredients.json"

# --- Helper Functions ---
def save_json(data, filename):
    """주어진 데이터를 JSON 파일로 저장"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"데이터를 '{filename}'에 저장했습니다.")
    except Exception as e:
        print(f"'{filename}' 저장 중 오류 발생: {e}")

def load_json(filename):
    """JSON 파일에서 데이터를 로드"""
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' 파일이 존재하지 않습니다.")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"'{filename}'에서 데이터를 로드했습니다.")
        return data
    except Exception as e:
        print(f"'{filename}' 로드 중 오류 발생: {e}")
        return None

# --- 모델 초기화 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
model = None
tokenizer = None

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"'{model_name}' 모델 및 토크나이저 로딩 완료.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    exit()

def run_llm_inference(chat_prompt, max_tokens=1024):
    """LLM 추론 실행"""
    try:
        input_ids = tokenizer.apply_chat_template(chat_prompt, return_tensors="pt", tokenize=True).to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.8,
            temperature=0.6,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    except Exception as e:
        print(f"LLM 추론 중 오류 발생: {e}")
        if "CUDA out of memory" in str(e):
            print("CUDA 메모리 부족! max_new_tokens를 줄이거나 모델 설정을 확인하세요.")
        return None

def extract_ingredients(llm_output):
    """레시피 텍스트에서 재료 목록 추출"""
    try:
        match = re.search(r"### 재료 목록 시작 ###\s*(.*?)\s*### 재료 목록 끝 ###", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            ingredient_block = match.group(1).strip()
            # 콤마 또는 줄바꿈 기준으로 분리
            lines = []
            for line in ingredient_block.split('\n'):
                for item in line.split(','):
                    item = item.strip()
                    if item:
                        lines.append(item)
            recipe_text = llm_output.replace(match.group(0), "").strip()
            return recipe_text, lines
    except Exception as e:
        print(f"재료 추출 중 오류 발생: {e}")
    return llm_output, []

def run_recipe_generation(query):
    """레시피 생성 및 재료 추출 실행"""
    prompt = f"""
사용자 요청: "{query}"

**작업 지시:**
1. 사용자 요청에 맞는 상세한 요리 레시피를 생성해주세요.
2. 레시피 생성 후, 명확하게 구분되는 섹션에 해당 레시피에 필요한 모든 재료 목록을 한 줄에 하나씩 나열해주세요.
   재료 목록 시작 전에 `### 재료 목록 시작 ###` 이라고 표시하고, 목록 끝에는 `### 재료 목록 끝 ###` 이라고 표시해주세요.
3. 재료는 가능한 한 구체적으로 작성해주세요. (예: '돼지고기' 대신 '돼지고기 목살', '김치' 대신 '배추김치')

**출력 예시:**
[레시피 내용]

### 재료 목록 시작 ###
배추김치 300g
돼지고기 목살 300g
두부 1모
대파 2대
양파 1개
### 재료 목록 끝 ###
"""

    chat = [
        {"role": "system", "content": "당신은 훌륭한 요리사입니다. 사용자의 요청에 따라 레시피를 만들고 필요한 재료 목록을 정확하게 추출하여 지정된 형식으로 제공해주세요."},
        {"role": "user", "content": prompt},
    ]

    # 레시피 생성
    recipe_response = run_llm_inference(chat, max_tokens=1500)
    if not recipe_response:
        return None

    # 재료 추출
    recipe_text, ingredients = extract_ingredients(recipe_response)
    
    # JSON 파일로 저장
    recipe_data = {
        "query": query,
        "recipe": recipe_text
    }
    ingredients_data = {
        "ingredients": ingredients
    }
    
    try:
        with open("recipes.json", "w", encoding="utf-8") as f:
            json.dump(recipe_data, f, ensure_ascii=False, indent=2)
        with open("ingredients.json", "w", encoding="utf-8") as f:
            json.dump(ingredients_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"JSON 저장 중 오류 발생: {e}")
    
    return {
        "recipe": recipe_text,
        "ingredients": ingredients
    }

def main():
    query = input("검색할 레시피 또는 질문을 입력하세요: ")
    # 1. 레시피 생성 및 재료 추출
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

    # 3. 재료 매칭 분석
    match_result = analyze_ingredients_matching(materials, recipe_data, ingredients_data)

    # 4. 결과 종합 및 출력
    print("=== 레시피 ===")
    print(recipe_info["recipe"])
    print("=== 재료 매칭 결과 ===")
    print("정확히 일치:", match_result["exact_matches"])
    print("유사:", match_result["similar_matches"])
    print("구할 수 없음:", match_result["unavailable"])

if __name__ == "__main__":
    main()