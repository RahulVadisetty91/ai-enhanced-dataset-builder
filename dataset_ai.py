import os
from llmware.dataset_tools import Datasets
from llmware.library import Library
from llmware.retrieval import Query
from llmware.parsers import Parser
from llmware.setup import Setup

def test_build_datasets():
    # Setup library
    library = Library().create_new_library("test_datasets")
    sample_files_path = Setup().load_sample_files()
    library.add_files(os.path.join(sample_files_path, "SmallLibrary"))
    library.generate_knowledge_graph()

    # Build dataset
    ds = Datasets(library)

    # Basic dataset for industry domain adaptation
    basic_embedding_ds = ds.build_text_ds(min_tokens=500, max_tokens=1000)
    assert basic_embedding_ds["testing_samples"] > 0

    # Self-supervised generative dataset - text and completion
    basic_generative_completion_ds = ds.build_gen_ds_targeted_text_completion(prompt_wrapper="alpaca")
    assert basic_generative_completion_ds["testing_samples"] > 0

    # Generative self-supervised training sets
    xsum_generative_completion_ds = ds.build_gen_ds_headline_text_xsum(prompt_wrapper="human_bot")
    assert xsum_generative_completion_ds["testing_samples"] > 0
    
    topic_prompter_ds = ds.build_gen_ds_headline_topic_prompter(prompt_wrapper="chat_gpt")
    assert topic_prompter_ds["testing_samples"] > 0

    # Filter library by a key term and build dataset
    filtered_ds = ds.build_text_ds(min_tokens=150, max_tokens=500, query="agreement", filter_dict={"master_index": 1})
    assert filtered_ds["testing_samples"] > 0

    # Create dataset from query results
    query_results = Query(library=library).query("salary")
    filtered_ds_from_query = ds.build_text_ds(min_tokens=250, max_tokens=600, qr=query_results)
    assert filtered_ds_from_query["testing_samples"] > 0

    # Images with text dataset
    images_with_text_ds = ds.build_visual_ds_image_labels()
    assert images_with_text_ds["testing_samples"] > 0

    library.delete_library()

def test_build_aws_transcribe_ds():
    # Setup library
    library = Library().create_new_library("aws_transcribe_ds")
    sample_files_path = Setup().load_sample_files()
    library.add_dialogs(os.path.join(sample_files_path, "AWS-Transcribe"))
    library.generate_knowledge_graph()

    ds = Datasets(library)

    # Generative conversation dataset
    generative_conversation_ds = ds.build_gen_dialog_ds(prompt_wrapper="human_bot", human_first=True)
    assert generative_conversation_ds["testing_samples"] > 0

    # Generative model fine-tuning dataset from prompt history
    generative_curated_ds = ds.build_gen_ds_from_prompt_history(prompt_wrapper="alpaca")
    assert generative_curated_ds["batches"] > 0

    library.delete_library()

# AI-Driven Enhancements
def build_advanced_dataset(ds, library, prompt_wrapper="chat_gpt"):
    """
    Build an advanced dataset using AI-driven analysis to refine dataset creation.
    This method integrates AI insights for enhanced dataset quality.
    """
    # AI analysis for improved dataset parameters
    ai_analysis = analyze_data_quality(library)
    advanced_ds = ds.build_text_ds(
        min_tokens=ai_analysis["min_tokens"],
        max_tokens=ai_analysis["max_tokens"],
        query=ai_analysis["query"]
    )
    return advanced_ds

def analyze_data_quality(library):
    """
    Placeholder function for AI-driven data quality analysis.
    This function should use machine learning models to analyze data and recommend improvements.
    """
    # Example AI analysis logic
    return {
        "min_tokens": 200,
        "max_tokens": 800,
        "query": "high_quality"
    }
