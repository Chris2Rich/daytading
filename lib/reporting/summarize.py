from typing import List
import dspy

def summarize(context: str=None, data: List[str]=None, use_md: bool=False):
    """
    Summarize a list of text data given a specific context.
    
    Args:
        context: The context around which the data should be summarized
        data: List of text strings to be summarized
        
    Returns:
        A markdown-formatted summary string
    """
    class Summarizer(dspy.Signature):
        """
        Convert large amounts of data into useful summaries of dense and relevant information.
        Return in dspy format.
        """
        context = dspy.InputField(desc="The context around which the data is to be summarized")
        data = dspy.InputField(desc="List of text data to summarize")
        if use_md:
            summary = dspy.OutputField(desc="Dense, relevant summary focused on the given context that is formatted in markdown")
        else:
            summary = dspy.OutputField(desc="Dense, relevant summary focused on the given context")
    
    summarizer = dspy.ChainOfThought(Summarizer)
    combined_data = "\n\n".join(data) if isinstance(data, list) else str(data)
    result = summarizer(context=context, data=combined_data)
    
    return result.summary