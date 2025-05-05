import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import numpy as np
import time
import json
from sklearn.preprocessing import MinMaxScaler
import re

def load_json_data():
    """JSON 파일들을 로드하고 분석에 필요한 데이터 준비"""
    try:
        with open("materials.json", "r", encoding="utf-8") as f:
            materials = json.load(f)
        with open("recipes.json", "r", encoding="utf-8") as f:
            recipe = json.load(f)
        with open("ingredients.json", "r", encoding="utf-8") as f:
            ingredients = json.load(f)
        return materials, recipe, ingredients
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return None, None, None

def extract_core_name(ingredient):
    # 수량/단위/약간/큰술/개/g/ml 등 제거
    return re.sub(r'\s*([0-9]+[a-zA-Z가-힣]*|약간|큰술|작은술|모|개|g|ml|L|컵|스푼|ts|tbsp|tsp|\(.*?\))$', '', ingredient).strip()

def lexical_similarity(a, b):
    set_a, set_b = set(a), set(b)
    jaccard = len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0
    import difflib
    lev = difflib.SequenceMatcher(None, a, b).ratio()
    return (jaccard + lev) / 2

def analyze_ingredients_matching(materials, recipe_data, ingredients_data):
    matching_result = {
        "exact_matches": [],
        "similar_matches": [],
        "unavailable": []
    }
    recipe_ingredients = ingredients_data.get("ingredients", [])
    core_ingredients = [extract_core_name(ing) for ing in recipe_ingredients]
    used_recipe_idx = set()
    mart_names_syns = [m[0] for m in materials]  # 동의어 리스트
    mart_locs = [m[1] for m in materials]
    # 1. exact match (동의어 중 하나라도 일치)
    for mart_idx, mart_syn_list in enumerate(mart_names_syns):
        for idx, core_ing in enumerate(core_ingredients):
            if any(mart_syn == core_ing for mart_syn in mart_syn_list):
                matching_result["exact_matches"].append({
                    "mart_item": mart_syn_list[0],  # 대표값
                    "location": mart_locs[mart_idx],
                    "ingredient": recipe_ingredients[idx],
                    "ingredient_idx": idx
                })
                used_recipe_idx.add(idx)
                break
    # 2. similar/unavailable match (동의어별 임베딩/유사도 평균)
    # bi-encoder 준비
    bi_models = [
        SentenceTransformer('jhgan/ko-sroberta-multitask'),
        SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    ]
    cross_models = [
        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512),
        CrossEncoder('bongsoo/kpf-cross-encoder-v1', max_length=512)
    ]
    # 동의어별로 임베딩/유사도 평균
    for idx, core_ing in enumerate(core_ingredients):
        if idx in used_recipe_idx:
            continue
        # bi-encoder 평균 유사도 계산
        bi_scores = []
        for mart_syn_list in mart_names_syns:
            syn_scores = []
            for model in bi_models:
                ing_emb = model.encode([core_ing], convert_to_tensor=True)[0]
                syn_embs = model.encode(mart_syn_list, convert_to_tensor=True)
                syn_scores.append(np.mean([float(util.cos_sim(ing_emb, syn_emb).item()) for syn_emb in syn_embs]))
            bi_scores.append(np.mean(syn_scores))
        bi_scores = np.array(bi_scores)
        top3 = np.argsort(bi_scores)[::-1][:3]
        # cross-encoder 평균 유사도 계산 (전체 후보)
        cross1_all = np.array([
            np.mean([cross_models[0].predict([(core_ing, syn)])[0] for syn in mart_names_syns[i]])
            for i in range(len(mart_names_syns))
        ])
        cross2_all = np.array([
            np.mean([cross_models[1].predict([(core_ing, syn)])[0] for syn in mart_names_syns[i]])
            for i in range(len(mart_names_syns))
        ])
        def minmax_01(x):
            if len(x) > 1 and x.max() != x.min():
                return (x - x.min()) / (x.max() - x.min())
            else:
                return np.array([0.0] * len(x))
        def minmax_signed(x):
            if len(x) > 1 and x.max() != x.min():
                return 2 * (x - x.min()) / (x.max() - x.min()) - 1
            else:
                return np.array([0.0] * len(x))
        cross1_norm_all = minmax_01(cross1_all)
        cross2_norm_all = minmax_signed(cross2_all)
        cross1_norm = [cross1_norm_all[tidx] for tidx in top3]
        cross2_norm = [cross2_norm_all[tidx] for tidx in top3]
        cross_avg_norm = [(c1 + c2) / 2 for c1, c2 in zip(cross1_norm, cross2_norm)]
        cross1_arr = [cross1_all[tidx] for tidx in top3]
        cross2_arr = [cross2_all[tidx] for tidx in top3]
        # lexical similarity: 동의어별로 계산 후 평균
        top3_lex_scores = [
            np.mean([lexical_similarity(core_ing, syn) for syn in mart_names_syns[tidx]])
            for tidx in top3
        ]
        top3_list = [
            (
                mart_names_syns[tidx][0],  # 대표값
                float(bi_scores[tidx]),
                float(cross1_arr[i]),
                float(cross2_arr[i]),
                float(cross1_norm[i]),
                float(cross2_norm[i]),
                float(cross_avg_norm[i]),
                float(top3_lex_scores[i]),
                float(0.4 * bi_scores[tidx] + 0.4 * cross_avg_norm[i] + 0.2 * top3_lex_scores[i]),
                mart_locs[tidx]
            )
            for i, tidx in enumerate(top3)
        ]
        best_idx = top3[0]
        best_bi_avg = bi_scores[best_idx]
        best_cross_avg_norm = cross_avg_norm[0]
        best_lex = top3_lex_scores[0]
        bi_norm = 0.5
        if len(bi_scores) > 1:
            bi_norm = (best_bi_avg - bi_scores.min()) / (bi_scores.max() - bi_scores.min() + 1e-8)
        combined_score = 0.4 * bi_norm + 0.4 * best_cross_avg_norm + 0.2 * best_lex
        print(f"[DEBUG] '{recipe_ingredients[idx]}'의 유사도 top3:")
        for rank, (name, bi_avg, cross1, cross2, cross1_n, cross2_n, cross_avg_n, lex, comb, loc) in enumerate(top3_list, 1):
            print(f"  {rank}. {name}: bi_avg={bi_avg:.3f}, cross1={cross1:.3f}, cross2={cross2:.3f}, cross1_norm={cross1_n:.3f}, cross2_norm={cross2_n:.3f}, cross_avg_norm={cross_avg_n:.3f}, lex={lex:.3f}, combined={comb:.3f} ({loc})")
        print(f"  [DEBUG] best_idx={best_idx}, bi_avg={best_bi_avg:.3f}, cross_avg_norm={best_cross_avg_norm:.3f}, lex={best_lex:.3f}, bi_norm={bi_norm:.3f}, combined={combined_score:.3f}")
        if combined_score >= 0.6:
            matching_result["similar_matches"].append({
                "mart_item": mart_names_syns[best_idx][0],
                "location": mart_locs[best_idx],
                "ingredient": recipe_ingredients[idx],
                "ingredient_idx": idx,
                "score": float(combined_score),
                "bi_similarity_avg": float(best_bi_avg),
                "cross_score_avg_norm": float(best_cross_avg_norm),
                "lexical_score": float(best_lex),
                "bi_similarity_normalized": float(bi_norm),
                "top3": top3_list
            })
            used_recipe_idx.add(idx)
        else:
            matching_result["unavailable"].append({
                "ingredient": recipe_ingredients[idx],
                "ingredient_idx": idx,
                "top3": top3_list,
                "bi_similarity_avg": float(best_bi_avg),
                "cross_score_avg_norm": float(best_cross_avg_norm),
                "lexical_score": float(best_lex),
                "bi_similarity_normalized": float(bi_norm),
                "combined_score": float(combined_score)
            })
    print("[DEBUG] 최종 similar_matches score 목록:")
    for item in matching_result["similar_matches"]:
        print(f"  {item['ingredient']} → {item['mart_item']} (score={item['score']:.3f}) bi_avg={item['bi_similarity_avg']:.3f} cross_avg_norm={item['cross_score_avg_norm']:.3f} lex={item['lexical_score']:.3f} bi_norm={item['bi_similarity_normalized']:.3f}")
    print("[DEBUG] 최종 unavailable score 목록:")
    for item in matching_result["unavailable"]:
        print(f"  {item['ingredient']} top3: {item['top3']} bi_avg={item['bi_similarity_avg']:.3f} cross_avg_norm={item['cross_score_avg_norm']:.3f} lex={item['lexical_score']:.3f} bi_norm={item['bi_similarity_normalized']:.3f} combined={item['combined_score']:.3f}")
    return matching_result

def combine_documents(materials_path="materials.json", recipes_path="recipes.json", ingredients_path="ingredients.json"):
    """세 가지 JSON 파일의 내용을 결합"""
    documents = []
    
    # 1. materials.json 로드
    try:
        with open(materials_path, "r", encoding="utf-8") as f:
            materials = json.load(f)
            for item in materials:
                documents.append(f"{item[0]}는 {item[1]} 구역에 있습니다.")
    except Exception as e:
        print(f"Error loading materials: {e}")
    
    # 2. recipes.json 로드
    try:
        with open(recipes_path, "r", encoding="utf-8") as f:
            recipe_data = json.load(f)
            if recipe_data and "recipe" in recipe_data:
                documents.append(recipe_data["recipe"])
    except Exception as e:
        print(f"Error loading recipe: {e}")
    
    # 3. ingredients.json 로드
    try:
        with open(ingredients_path, "r", encoding="utf-8") as f:
            ingredients_data = json.load(f)
            if ingredients_data and "ingredients" in ingredients_data:
                for ingredient in ingredients_data["ingredients"]:
                    documents.append(f"레시피에 {ingredient}이(가) 필요합니다.")
    except Exception as e:
        print(f"Error loading ingredients: {e}")
    
    return documents

def get_final_top_doc_content(query="김치찌개 레시피를 알려주고 재료들이 어디에 있는지도 알려줘", 
                            materials_path="materials.json"):
    """
    기존 RAG 검색에 재료 매칭 분석을 추가한 버전
    """
    # 1. 데이터 로드 및 재료 분석
    materials, recipe, ingredients = load_json_data()
    if not materials:
        print("Error: Could not load required data")
        return []

    # 2. 재료 매칭 분석 수행
    matching_results = analyze_ingredients_matching(materials, recipe, ingredients)
    
    # 3. 문서 준비
    documents = []
    
    # 마트 재료 정보
    for item in materials:
        documents.append(f"{item[0]}는 {item[1]} 구역에 있습니다.")
    
    # 레시피 정보
    if recipe and "recipe" in recipe:
        documents.append(recipe["recipe"])
    
    # 재료 매칭 결과 추가
    if matching_results:
        # 정확히 일치하는 재료
        for match in matching_results["exact_matches"]:
            documents.append(f"{match['ingredient']}는 {match['location']}에서 찾을 수 있습니다.")
        
        # 유사한 재료
        for match in matching_results["similar_matches"]:
            documents.append(
                f"{match['ingredient']} 대신 {match['mart_item']}를 "
                f"{match['location']}에서 찾을 수 있습니다."
            )
        
        # 구할 수 없는 재료
        if matching_results["unavailable"]:
            documents.append(
                f"다음 재료들은 마트에서 찾을 수 없습니다: "
                f"{', '.join([un['ingredient'] for un in matching_results['unavailable']])}"
            )

    # 4. 기존 RAG 로직 수행
    if not documents:
        print("Error: No documents found.")
        return []

    # --- 설정값 ---
    use_gpu = torch.cuda.is_available()
    default_device = 'cuda' if use_gpu else 'cpu'
    print(f"Using device: {default_device}")

    top_k_retrieval = 30  # Bi-Encoder로 검색할 초기 후보 수
    top_k_final = 10      # 최종 반환할 문서 수

    if top_k_final > top_k_retrieval:
        print(f"Warning: top_k_final({top_k_final}) > top_k_retrieval({top_k_retrieval}). Setting top_k_final = top_k_retrieval.")
        top_k_final = top_k_retrieval

    # --- 2. 초기 검색 (Bi-Encoder + FAISS-CPU) ---
    print("\n--- Initial Retrieval (Bi-Encoder + FAISS) ---")
    start_time_bi = time.time()
    try:
        bi_encoder = SentenceTransformer('jhgan/ko-sroberta-multitask', device=default_device)

        doc_embeddings = bi_encoder.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True, # IP (Inner Product)는 정규화된 벡터에서 코사인 유사도와 동일
            show_progress_bar=True
        )
        embedding_dim = doc_embeddings.shape[1]

        # FAISS 인덱스 생성 및 추가
        index_cpu = faiss.IndexFlatIP(embedding_dim) # Inner Product 사용
        faiss_index = index_cpu

        faiss_index.add(doc_embeddings)

        # 쿼리 임베딩 및 검색
        query_embedding = bi_encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding_reshaped = np.expand_dims(query_embedding, axis=0)

        similarities, indices = faiss_index.search(query_embedding_reshaped, k=top_k_retrieval)

        retrieved_indices = indices[0]
        retrieved_similarities = similarities[0] # Bi-Encoder scores (cosine similarity)

        # 검색 결과 저장
        retrieved_docs_info = []
        for i, idx in enumerate(retrieved_indices):
            if idx != -1: # 유효한 인덱스인지 확인
                similarity = retrieved_similarities[i]
                if similarity < -1.0 or similarity > 1.0:
                     similarity = np.clip(similarity, -1.0, 1.0)

                doc_content = documents[idx]
                retrieved_docs_info.append({'index': idx, 'bi_similarity': similarity, 'content': doc_content})

    except Exception as e:
        print(f"Error during Bi-Encoder retrieval: {e}")
        return []

    end_time_bi = time.time()
    print(f"Bi-Encoder retrieval completed in {end_time_bi - start_time_bi:.2f} seconds.")

    if not retrieved_docs_info:
        print("No documents retrieved by Bi-Encoder.")
        return []

    print(f"\n초기 검색 결과 (Bi-Encoder, 상위 {len(retrieved_docs_info)}개 중 일부):")
    for i in range(min(5, len(retrieved_docs_info))): # 처음 5개만 출력
        doc = retrieved_docs_info[i]
        print(f"- 인덱스: {doc['index']}, Bi-Sim: {doc['bi_similarity']:.4f}, 내용: {doc['content'][:50]}...")

    # --- 3. 재순위화 점수 계산 (Cross-Encoder) ---
    print("\n--- Reranking Score Calculation (Cross-Encoder) ---")
    start_time_cross = time.time()
    try:
        cross_encoder = CrossEncoder('bongsoo/kpf-cross-encoder-v1', max_length=512, device=default_device)

        rerank_candidate_pairs = [(query, doc['content']) for doc in retrieved_docs_info]

        cross_scores = cross_encoder.predict(
            rerank_candidate_pairs,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    except Exception as e:
        print(f"Error during Cross-Encoder prediction: {e}")
        return []

    end_time_cross = time.time()
    print(f"Cross-Encoder prediction completed in {end_time_cross - start_time_cross:.2f} seconds.")

    # --- 4. 점수 결합 및 최종 순위 결정 ---
    print("\n--- Combining Scores and Final Ranking ---")

    scaler_bi = MinMaxScaler()
    scaler_cross = MinMaxScaler()

    bi_similarities = np.array([doc['bi_similarity'] for doc in retrieved_docs_info]).reshape(-1, 1)
    bi_similarities = np.nan_to_num(bi_similarities)

    cross_scores_reshaped = cross_scores.reshape(-1, 1)
    cross_scores_reshaped = np.nan_to_num(cross_scores_reshaped)

    if len(bi_similarities) > 1:
        normalized_bi_scores = scaler_bi.fit_transform(bi_similarities).flatten()
    else:
        normalized_bi_scores = np.array([0.5] * len(bi_similarities))

    if len(cross_scores_reshaped) > 1:
        normalized_cross_scores = scaler_cross.fit_transform(cross_scores_reshaped).flatten()
    else:
        normalized_cross_scores = np.array([0.5] * len(cross_scores_reshaped))

    combined_results_info = []
    for i, doc_info in enumerate(retrieved_docs_info):
        norm_bi = normalized_bi_scores[i]
        norm_cross = normalized_cross_scores[i]
        combined_score = (norm_bi + norm_cross) / 2.0

        combined_results_info.append({
            'index': doc_info['index'],
            'content': doc_info['content'],
            'bi_similarity_original': doc_info['bi_similarity'],
            'cross_score_original': float(cross_scores[i]),
            'bi_similarity_normalized': norm_bi,
            'cross_score_normalized': norm_cross,
            'combined_score': combined_score
        })

    sorted_combined_results = sorted(
        combined_results_info,
        key=lambda x: x['combined_score'],
        reverse=True
    )

    final_top_docs_info = sorted_combined_results[:top_k_final]
    print(f"\n최종 선택된 문서들 (Combined Score, 상위 {top_k_final}개):")
    final_return_list = []
    for i, doc in enumerate(final_top_docs_info):
        print(f"Rank {i+1}:")
        print(f"  - 인덱스: {doc['index']}")
        print(f"  - Combined Score: {doc['combined_score']:.4f}")
        print(f"  - Bi-Sim (Norm): {doc['bi_similarity_normalized']:.4f} (Orig: {doc['bi_similarity_original']:.4f})")
        print(f"  - Cross-Score (Norm): {doc['cross_score_normalized']:.4f} (Orig: {doc['cross_score_original']:.4f})")
        print(f"  - 내용: {doc['content'][:80]}...")
        final_return_list.append((doc['index'], doc['combined_score'], doc['content']))

    if not final_return_list:
        print("No documents selected after combining scores.")
        return []

    return final_return_list
#get_final_top_doc_content()