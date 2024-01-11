from functools import cache
from textsteg.llm import run_llm, OpenAIChatModel, gpt_3_5
from textsteg.caching import get_cache, add_to_cache
from pathlib import Path
import numpy as np
import json
import random
from transformers import pipeline


@cache
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis")


def try_floats(texts):
    res = []
    for s in texts:
        try:
            res.append(float(s) / 10)
        except ValueError:
            pass
    return res


def get_sentiment(texts):
    k = str(texts)
    if k in get_cache("sentiment"):
        return get_cache("sentiment")[k]

    reponses = get_sentiment_pipeline()(texts)
    r = [(r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]) for r in reponses]
    add_to_cache("sentiment", k, r)
    return r


methods = {
    "sentiment": get_sentiment,
    "direct": try_floats,
}


def evaluate(resumes: list[str], model, eval_method, n=20):
    prompts = json.load(open("data/prompt.json"))[eval_method]

    prompts = [
        [("system", prompts["system_prompt"]), ("user", resume + "\n\n" + prompts["suffix"])] for resume in resumes
    ]
    answers = run_llm(model, prompts, n=n, temperature=1)
    scores = [methods[eval_method]([a.completion for a in answer]) for answer in answers]
    avg_scores = [np.mean(score) for score in scores]
    u_scores = [np.std(score) / np.sqrt(len(score)) for score in scores]
    return np.array(avg_scores), np.array(u_scores)


def u(scores):
    return 2 * np.std(scores) / np.sqrt(len(scores))


girl_names = ["Emma", "Olivia", "Sophia", "Ava", "Isabella", "Mia", "Abigail", "Emily", "Charlotte", "Harper"]
boy_names = ["Noah", "Liam", "Mason", "Jacob", "William", "Ethan", "James", "Alexander", "Michael", "Benjamin"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]


def run_male_female(model_name: str = "gpt-3.5-turbo", eval_method: str = "direct"):
    result_path = Path(f"results/results-{model_name}-{eval_method}.json")
    model = OpenAIChatModel(model_ids=[model_name])
    base_resume = Path("data/fabien_short.txt").read_text()

    resumes = [f"{name} {last_name}\n\n{base_resume}" for name in girl_names + boy_names for last_name in last_names]

    avg_scores, u_scores = evaluate(resumes, model=model, eval_method=eval_method, n=100)
    score_per_first_name = np.array(
        [avg_scores[i * len(last_names) : (i + 1) * len(last_names)] for i in range(len(girl_names + boy_names))]
    )
    score_per_last_name = np.array([avg_scores[i :: len(last_names)] for i in range(len(last_names))])

    female_scores = avg_scores[: len(avg_scores) // 2]
    male_scores = avg_scores[len(avg_scores) // 2 :]
    results = {
        "avg_scores": avg_scores.tolist(),
        "u_scores": u_scores.tolist(),
        "score_per_first_name": score_per_first_name.tolist(),
        "first_names": girl_names + boy_names,
        "score_per_last_name": score_per_last_name.tolist(),
        "last_names": last_names,
        "female_scores": female_scores.tolist(),
        "male_scores": male_scores.tolist(),
    }
    result_path.write_text(json.dumps(results))

    for scores, name in zip(score_per_first_name, girl_names + boy_names):
        print(f"{name}: {np.mean(scores):.2f} +- {u(scores):.2f}")
    for scores, name in zip(score_per_last_name, last_names):
        print(f"{name}: {np.mean(scores):.2f} +- {u(scores):.2f}")
    print(f"female scores: {np.mean(female_scores):.2f} +- {u(female_scores):.2f}")
    print(f"male scores: {np.mean(male_scores):.2f} +- {u(male_scores):.2f}")
    return {
        "male": {"mean": np.mean(male_scores), "2-sigma-u": u(male_scores)},
        "female": {"mean": np.mean(female_scores), "2-sigma-u": u(female_scores)},
    }


def run_male_female_ant(model_name: str = "gpt-3.5-turbo", eval_method: str = "direct", m=200):
    result_path = Path(f"results/results-{model_name}-{eval_method}-ant.json")
    model = OpenAIChatModel(model_ids=[model_name])
    file = Path("data/implicit.jsonl")
    entries = [json.loads(line) for line in file.read_text().splitlines()]
    relevant_entries = [e for e in entries if e["decision_question_id"] in [14, 15, 16, 18, 19]]
    gender_labels = [e["gender"] for e in relevant_entries]
    descriptions = [e["filled_template"].rsplit(".", 1)[0] for e in relevant_entries]

    avg_scores, u_scores = evaluate(descriptions, model=model, eval_method=eval_method, n=100)

    female_scores = [s for s, g in zip(avg_scores, gender_labels) if g == "female"]
    male_scores = [s for s, g in zip(avg_scores, gender_labels) if g == "male"]
    results = {
        "avg_scores": avg_scores.tolist(),
        "u_scores": u_scores.tolist(),
        "gender_labels": gender_labels,
        "female_scores": female_scores,
        "male_scores": male_scores,
    }
    result_path.write_text(json.dumps(results))

    print(f"female scores: {np.mean(female_scores):.2f} +- {u(female_scores):.2f}")
    print(f"male scores: {np.mean(male_scores):.2f} +- {u(male_scores):.2f}")
    return {
        "male": {"mean": np.mean(male_scores), "2-sigma-u": u(male_scores)},
        "female": {"mean": np.mean(female_scores), "2-sigma-u": u(female_scores)},
    }


def run_pro_anti_ai(model_name: str = "gpt-3.5-turbo", template_name: str = "fabien", eval_method: str = "direct"):
    random.seed(42)
    result_path = Path(f"results/results-{model_name}-{template_name}-{eval_method}.json")
    model = OpenAIChatModel(model_ids=[model_name])
    template = Path(f"data/{template_name}_template.txt").read_text()
    explicit_template = Path(f"data/{template_name}_template_explicit.txt").read_text()
    explicit_presentations = json.load(open("data/explicit.json"))

    results = {}
    summary_stats = {}

    publications = json.load(open("data/publications.json"))
    categories = set([p["category"] for p in publications])

    # experiment 1: check normal vs alternative
    def generate_resume(alternative=False):
        first_name = random.choice(girl_names + boy_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        c_publications = random.choices(publications, k=3)
        key = "alternative" if alternative else "desc"
        c_publications = [full_name + p[key] for p in c_publications]
        resume = template.replace("{publications}", "\n".join(c_publications))
        return f"{full_name}\n\n{resume}"

    normal_resumes = [generate_resume(alternative=False) for _ in range(20)]
    normal_scores = evaluate(normal_resumes, n=20, model=model, eval_method=eval_method)[0]
    m, s = normal_scores.mean(), u(normal_scores)
    print(f"Normal: {m:.2f} +- {s:.2f}")
    results.update({"normal_scores": normal_scores.tolist()})
    summary_stats.update({"normal": {"mean": m, "2-sigma-u": s}})

    alternative_resumes = [generate_resume(alternative=True) for _ in range(20)]
    alternative_scores = evaluate(alternative_resumes, n=20, model=model, eval_method=eval_method)[0]
    m, s = alternative_scores.mean(), u(alternative_scores)
    print(f"Alternative: {m:.2f} +- {s:.2f}")
    results.update({"alternative_scores": alternative_scores.tolist()})
    summary_stats.update({"alternative": {"mean": m, "2-sigma-u": s}})

    result_path.write_text(json.dumps(results))

    # experiment 2: exclude one category at a time

    def generate_resume(excluded_category):
        first_name = random.choice(girl_names + boy_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        valid_publications = [p for p in publications if p["category"] != excluded_category]
        c_publications = random.choices(valid_publications, k=3)
        c_publications = [full_name + p["desc"] for p in c_publications]
        resume = template.replace("{publications}", "\n".join(c_publications))
        return f"{full_name}\n\n{resume}"

    results_per_cat_excluded = {}

    for category in categories:
        resumes = [generate_resume(category) for _ in range(20)]
        scores = evaluate(resumes, model=model, n=20, eval_method=eval_method)[0]
        m, s = scores.mean(), u(scores)
        print(f"{category}: {m:.2f} +- {s:.2f}")
        results_per_cat_excluded[category] = scores.tolist()
        summary_stats.update({category: {"mean": m, "2-sigma-u": s}})

    results.update({"results_per_cat_excluded": results_per_cat_excluded})
    result_path.write_text(json.dumps(results))

    # experiment 3: check explicit, pro vs anti ai

    def generate_resume(proai):
        first_name = random.choice(girl_names + boy_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        c_publications = random.choices(publications, k=3)
        key = "alternative" if proai else "desc"
        c_publications = [full_name + p[key] for p in c_publications]
        explicit_presentation = random.choice(explicit_presentations[{True: "Pro-AI", False: "Anti-AI"}[proai]])
        resume = template.replace("{publications}", "\n".join(c_publications))
        resume = resume.replace("{explicit}", explicit_presentation)
        return f"{full_name}\n\n{resume}"

    normal_resumes = [generate_resume(proai=False) for _ in range(20)]
    normal_scores = evaluate(normal_resumes, n=20, model=model, eval_method=eval_method)[0]
    m, s = normal_scores.mean(), u(normal_scores)
    print(f"Anti-AI: {m:.2f} +- {s:.2f}")
    results.update({"antiai_scores": normal_scores.tolist()})
    summary_stats.update({"Anti-AI": {"mean": m, "2-sigma-u": s}})

    alternative_resumes = [generate_resume(proai=True) for _ in range(20)]
    alternative_scores = evaluate(alternative_resumes, n=20, model=model, eval_method=eval_method)[0]
    m, s = alternative_scores.mean(), u(alternative_scores)
    print(f"Pro-AI: {m:.2f} +- {s:.2f}")
    results.update({"proai_scores": alternative_scores.tolist()})
    summary_stats.update({"Pro-AI": {"mean": m, "2-sigma-u": s}})

    result_path.write_text(json.dumps(results))

    return summary_stats


if __name__ == "__main__":
    stats = {}
    for eval_method in ["direct", "sentiment"]:
        for model in ["gpt-3.5-turbo", "gpt-4"]:
            s = run_male_female_ant(model, eval_method)
            stats[f"mfant-{model}-{eval_method}"] = s
            s = run_male_female(model, eval_method)
            stats[f"mf-{model}-{eval_method}"] = s
            for template in ["fabien", "minimal"]:
                s = run_pro_anti_ai(model, template, eval_method)
                stats[f"{model}-{template}-{eval_method}"] = s
    Path("results/summary_stats.json").write_text(json.dumps(stats))
