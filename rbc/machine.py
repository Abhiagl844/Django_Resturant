import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score , ConfusionMatrixDisplay
from itemCart.models import orderedItems , order1
from .models import food , Review
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter , defaultdict
import json
from django.db.models.query import QuerySet

def dict_to_vector(ingre_dict, all_ingre):
    vector = []
    for ing in all_ingre:
        val = ingre_dict.get(ing, 0)
        try:
            vector.append(float(val))
        except (ValueError, TypeError):
            vector.append(0.0)
    return np.array(vector)


# -----------------------------
# Get all unique ingredients
# -----------------------------
def get_all_ingre():
    all_ingre = set()

    for item in food.objects.all():
        if not item.ingredients:
            continue

        if isinstance(item.ingredients, str):
            try:
                item.ingredients = json.loads(item.ingredients)
            except json.JSONDecodeError:
                continue

        all_ingre.update(item.ingredients.keys())

    return list(all_ingre)


# -----------------------------
# Similarity score between two food items
# -----------------------------
def get_similarity_score(item1, item2, all_ingre):
    score = 0

    if item1.course == item2.course:
        score += 1

    if item1.type == item2.type:
        score += 1

    if not item1.ingredients or not item2.ingredients:
        return score

    vect1 = dict_to_vector(item1.ingredients, all_ingre)
    vect2 = dict_to_vector(item2.ingredients, all_ingre)

    if vect1.sum() == 0 or vect2.sum() == 0:
        ingre_similar = 0
    else:
        ingre_similar = cosine_similarity([vect1], [vect2])[0][0]

    score += ingre_similar * 3
    return score


# -----------------------------
# Frequency-based recommendation
# -----------------------------
def recommended_freq():
    top_n = 5
    order_items = orderedItems.objects.all()

    f_counter = Counter()
    for item in order_items:
        f_counter[item.item] += item.quantity

    freq_items = [food_item.name for food_item, _ in f_counter.most_common(top_n)]
    return freq_items


# -----------------------------
# Content-based recommendation
# -----------------------------
def recommend_content_based(req):
    if orderedItems.objects.filter(order__user=req.user).exists():
        top_n = 5

        ordered_item_ids = (
            orderedItems.objects
            .filter(order__user=req.user)
            .values_list('item', flat=True)
            .distinct()
        )

        user_items = food.objects.filter(id__in=ordered_item_ids)
        all_items = food.objects.exclude(id__in=ordered_item_ids)

        scores = defaultdict(float)
        all_ingre = get_all_ingre()

        for user_item in user_items:
            for other_item in all_items:
                scores[other_item] += get_similarity_score(
                    user_item, other_item, all_ingre
                )

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = [item for item, _ in sorted_items[:top_n]]

        return result if result else recommended_freq()

    return recommended_freq()


# -----------------------------
# Final suggestions used in views
# -----------------------------
def seggestions(req):
    if req.user.is_authenticated:
        items = recommend_content_based(req)
    else:
        items = recommended_freq()

    result = []
    for i in items:
        if isinstance(i, food):
            result.append(i)
        else:
            try:
                result.append(food.objects.get(name=i))
            except food.DoesNotExist:
                pass

    return result