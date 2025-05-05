from typing import List, Tuple, Dict
import re
import json

def extract_materials_from_rag_result(content: str) -> List[Tuple[str, str]]:
    """
    RAG 결과 텍스트에서 재료와 위치 정보를 추출
    Returns: List of (material_name, location) tuples
    """
    if not content:
        return []

    # 재료-위치 패턴 검색을 위한 정규표현식
    patterns = [
        r"([가-힣a-zA-Z0-9]+)(?:는|은|이|가)\s*([A-Z]-\d+)\s*구역",  # 기본 패턴
        r"([가-힣a-zA-Z0-9]+):\s*([A-Z]-\d+)",                      # 콜론 구분
        r"([가-힣a-zA-Z0-9]+)\s*\(([A-Z]-\d+)\)",                  # 괄호 구분
        r"([A-Z]-\d+)(?:\s*구역)?(?:에|에서)?\s*([가-힣a-zA-Z0-9]+)", # 위치 먼저 나오는 패턴
    ]

    materials = []
    seen = set()  # 중복 방지

    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            # 패턴에 따라 그룹 순서 조정
            if pattern == patterns[-1]:  # 위치가 먼저 나오는 패턴
                location, material = match.groups()
            else:
                material, location = match.groups()
            
            # 재료명과 위치 정보 정제
            material = material.strip()
            location = location.strip()
            
            # 유효성 검사
            if (not material or not location or 
                not re.match(r'^[A-Z]-\d+$', location) or
                (material, location) in seen):
                continue
                
            materials.append((material, location))
            seen.add((material, location))

    return materials

def analyze_material_matches(recipe_ingredients: List[str], 
                          mart_materials: List[Tuple[str, str]]) -> Dict:
    """
    레시피 재료와 마트 재료 목록을 비교하여 매칭 분석
    Returns: Dictionary with available, unavailable, and similar materials
    """
    mart_dict = dict(mart_materials)
    available = {}
    unavailable = []
    similar = {}
    
    for ingredient in recipe_ingredients:
        ingredient_lower = ingredient.lower()
        found = False
        
        # 정확히 일치하는 재료 찾기
        for mart_item, location in mart_dict.items():
            if mart_item.lower() == ingredient_lower:
                available[ingredient] = location
                found = True
                break
        
        if not found:
            # 유사한 재료 찾기
            for mart_item, location in mart_dict.items():
                # 재료 카테고리별 매칭 규칙
                categories = {
                    "고기류": ["고기", "살", "갈비", "안심", "등심"],
                    "채소류": ["채소", "파", "배추", "무", "당근"],
                    "김치류": ["김치", "깍두기", "무김치"],
                    "양념류": ["고추장", "된장", "간장", "소스"],
                    "해산물": ["생선", "새우", "조개", "멸치"],
                }
                
                for category_items in categories.values():
                    if any(keyword in ingredient_lower for keyword in category_items) and \
                       any(keyword in mart_item.lower() for keyword in category_items):
                        similar[f"{mart_item} ({location})"] = ingredient
                        found = True
                        break
                if found:
                    break
            
            if not found:
                unavailable.append(ingredient)
    
    return {
        "available": available,
        "unavailable": unavailable,
        "similar": similar
    }

def find_location(ingredient):
    with open("materials.json", "r", encoding="utf-8") as f:
        materials = json.load(f)
    for mart_syn_list, location in materials:
        if ingredient in mart_syn_list:
            return mart_syn_list[0], location  # 대표값, 위치
    return None, None

# 사용 예시 (테스트용)
if __name__ == "__main__":
    test_ingredients = ["마늘", "마늘쫑", "갈릭", "양파", "없는재료"]
    for ing in test_ingredients:
        name, loc = find_location(ing)
        if name:
            print(f"{ing}는 {name}({loc}) 구역에서 찾을 수 있습니다.")
        else:
            print(f"{ing}는 마트에서 찾을 수 없습니다.")
