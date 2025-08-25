import json
from location import extract_materials_from_rag_result
from RAG_test import analyze_ingredients_matching
import http.client

# --- RAG Reasoning API 호출 ---
def call_rag_reasoning_api(query, api_key, request_id=None):
    host = 'clovastudio.stream.ntruss.com'
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': api_key
    }
    if request_id:
        headers['X-NCP-CLOVASTUDIO-REQUEST-ID'] = request_id

    tools = [
        {
            "function": {
                "description": "레시피 및 마트 재료 검색 도구입니다.",
                "name": "recipe_mart_retrieval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "요리명 또는 질문"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    request_data = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "tools": tools,
        "toolChoice": "auto",
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 1024,
        "temperature": 0.5,
        "repetitionPenalty": 1.1,
        "stop": [],
        "seed": 0,
        "includeAiFilters": True
    }
    conn = http.client.HTTPSConnection(host)
    conn.request('POST', '/v1/api-tools/rag-reasoning', json.dumps(request_data), headers)
    response = conn.getresponse()
    result = json.loads(response.read().decode('utf-8'))
    conn.close()
    return result

# --- Skillset API(최종 답변) 호출 ---
def call_skillset_final_answer_api(query, api_key, skillset_id, version, chat_history=None, request_id=None):
    host = 'clovastudio.stream.ntruss.com'
    path = f'/v1/skillsets/{skillset_id}/versions/{version}/final-answer'
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': api_key
    }
    if request_id:
        headers['X-NCP-CLOVASTUDIO-REQUEST-ID'] = request_id

    body = {
        "query": query,
        "tokenStream": False
    }
    if chat_history:
        body["chatHistory"] = chat_history

    conn = http.client.HTTPSConnection(host)
    conn.request('POST', path, json.dumps(body), headers)
    response = conn.getresponse()
    result = json.loads(response.read().decode('utf-8'))
    conn.close()
    return result

def extract_ingredients_from_api_response(api_message_content):
    """
    RAG Reasoning API 또는 Skillset API의 응답에서 재료 목록 추출
    기존 extract_ingredients 함수와 유사하게, '### 재료 목록 시작 ### ... ### 재료 목록 끝 ###' 패턴 사용
    """
    import re
    match = re.search(r"### 재료 목록 시작 ###\s*(.*?)\s*### 재료 목록 끝 ###", api_message_content, re.DOTALL | re.IGNORECASE)
    if match:
        ingredient_block = match.group(1).strip()
        lines = []
        for line in ingredient_block.split('\n'):
            for item in line.split(','):
                item = item.strip()
                if item:
                    lines.append(item)
        recipe_text = api_message_content.replace(match.group(0), "").strip()
        return recipe_text, lines
    return api_message_content, []

def main():
    # 실제 API 키, skillset-id, version 입력 필요
    API_KEY = 'Bearer nv-e84b8280d6b14f4a93b7302170925f5bEnGX'
    SKILLSET_ID = '<skillset-id>'
    VERSION = 1

    query = input("검색할 레시피 또는 질문을 입력하세요: ")

    # 1. RAG Reasoning API로 도구 사용 계획 세우기
    rag_result = call_rag_reasoning_api(query, API_KEY)
    if rag_result.get("status", {}).get("code") != "20000":
        print("RAG Reasoning API 오류:", rag_result)
        return
    message = rag_result["result"]["message"]
    print("\n=== RAG Reasoning 결과 ===")
    print("사고 과정:", message.get("thinkingContent", ""))
    print("도구 호출 계획:", json.dumps(message.get("toolCalls", []), ensure_ascii=False, indent=2))

    # 2. Skillset API로 실제 도구 실행 및 최종 답변 생성
    skillset_result = call_skillset_final_answer_api(query, API_KEY, SKILLSET_ID, VERSION)
    if skillset_result.get("status", {}).get("code") != "20000":
        print("Skillset API 오류:", skillset_result)
        return
    result = skillset_result["result"]
    print("\n=== 최종 답변 결과 ===")
    print("최종 답변:", result.get("finalAnswer", ""))
    print("API 호출 내역:", json.dumps(result.get("apiResult", []), ensure_ascii=False, indent=2))

    # 3. 레시피 및 재료 추출 (최종 답변에서 추출)
    recipe_text, ingredients = extract_ingredients_from_api_response(result.get("finalAnswer", ""))
    recipe_data = {
        "query": query,
        "recipe": recipe_text
    }
    ingredients_data = {
        "ingredients": ingredients
    }
    # 파일 저장 (기존 파이프라인과 호환)
    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(recipe_data, f, ensure_ascii=False, indent=2)
    with open("ingredients.json", "w", encoding="utf-8") as f:
        json.dump(ingredients_data, f, ensure_ascii=False, indent=2)

    # 4. 파일 로드 및 재료 매칭 분석 (기존 함수 그대로 활용)
    with open("materials.json", "r", encoding="utf-8") as f:
        materials = json.load(f)
    match_result = analyze_ingredients_matching(materials, recipe_data, ingredients_data)

    # 5. 결과 종합 및 출력 (기존 방식 유지)
    print("\n=== 레시피 ===")
    print(recipe_text)
    print("\n=== 재료 매칭 결과 ===")
    print("정확히 일치:")
    for item in match_result["exact_matches"]:
        print(f"- {item['ingredient']}: {item['location']}")
    print("유사:")
    for item in match_result["similar_matches"]:
        print(f"- {item['ingredient']} → {item['mart_item']} ({item['location']})")
    print("구할 수 없음:")
    for item in match_result["unavailable"]:
        print(f"- {item['ingredient']}")

if __name__ == "__main__":
    main()