import textwrap
from IPython.display import Markdown

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def get_performance_category(score, grading_system):
    """
    Categorize performance based on the score and the grading system.

    Args:
    - score (float): The score to categorize.
    - grading_system (str): The grading system to use ('percentage', 'pass_fail', 'abcdf').

    Returns:
    - str: The performance category.
    """
    if grading_system == "percentage":
        percentage = score * 100
        if percentage >= 70:
            return "Good"
        elif percentage < 50:
            return "Needs Improvement"
        else:
            return "Satisfactory"
    elif grading_system == "pass_fail":
        return "Pass" if score >= 0.5 else "Fail"
    elif grading_system == "abcdf":
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    else:
        raise ValueError(f"Unknown grading system: {grading_system}")

def get_prompt(cluster_data, cluster_label, section_mapping, grading_system):
    """
    Generate a prompt for the language model to create explanations.

    Args:
    - cluster_data (dict): Data of the cluster to generate explanations for.
    - cluster_label (int): The label of the cluster.
    - section_mapping (dict): Mapping of sections to descriptive names.
    - grading_system (str): The grading system to use ('percentage', 'pass_fail', 'abcdf').

    Returns:
    - list: The prompt to be used by the language model.
    """
    if grading_system == "percentage":
        sections_performance = {
            "Good": [],
            "Satisfactory": [],
            "Needs Improvement": []
        }
    elif grading_system == "pass_fail":
        sections_performance = {
            "Pass": [],
            "Fail": []
        }
    elif grading_system == "abcdf":
        sections_performance = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
            "F": []
        }
    else:
        raise ValueError(f"Unknown grading system: {grading_system}")

    for section, score in cluster_data.items():
        performance_category = get_performance_category(score, grading_system)
        if performance_category in sections_performance:
            sections_performance[performance_category].append(section_mapping[section])

    prompt = [
        f"Cluster {cluster_label} performance breakdown:"
    ]

    if grading_system == "percentage":
        if sections_performance["Good"]:
            prompt.extend([
                "Good performance in sections (Strengths):",
                "\n".join(sections_performance["Good"])
            ])
        else:
            prompt.append("Good performance in sections (Strengths): None")

        if sections_performance["Satisfactory"]:
            prompt.extend([
                "Satisfactory performance in sections (Weaknesses):",
                "\n".join(sections_performance["Satisfactory"])
            ])
        else:
            prompt.append("Satisfactory performance in sections (Weaknesses): None")

        if sections_performance["Needs Improvement"]:
            prompt.extend([
                "Low performance in sections (Weaknesses):",
                "\n".join(sections_performance["Needs Improvement"])
            ])
        else:
            prompt.append("Low performance in sections (Weaknesses): None")
    elif grading_system == "pass_fail":
        if sections_performance["Pass"]:
            prompt.extend([
                "Passed sections:",
                "\n".join(sections_performance["Pass"])
            ])
        else:
            prompt.append("Passed sections: None")

        if sections_performance["Fail"]:
            prompt.extend([
                "Failed sections:",
                "\n".join(sections_performance["Fail"])
            ])
        else:
            prompt.append("Failed sections: None")
    elif grading_system == "abcdf":
        for grade in ["A", "B", "C", "D", "F"]:
            if sections_performance[grade]:
                prompt.extend([
                    f"Sections with grade {grade}:",
                    "\n".join(sections_performance[grade])
                ])
            else:
                prompt.append(f"Sections with grade {grade}: None")
    else:
        raise ValueError(f"Unknown grading system: {grading_system}")

    prompt.append("Please provide explanations for the strengths and weaknesses of this cluster's performance in Machine Learning Examination, along with recommendations for improvement.")

    return prompt
