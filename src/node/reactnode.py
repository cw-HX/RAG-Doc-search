from src.state.rag_state import RAGState

class RAGNodes:
    """LangGraph nodes for Architectural Codebase Exploration"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(
            f"File: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" 
            for doc in state.retrieved_docs
        )

        # Step 1: Generate architectural explanation
        answer_prompt = f"""
        You are an Expert Software Architect. Analyze the following code context and answer the user's question.
        Focus on structural relationships, class hierarchies, and data flow.

        Context:
        {context}

        Question:
        {state.question}
        """
        answer_response = self.llm.invoke(answer_prompt)

        # Step 2: Generate Mermaid.js Diagram
        diagram_prompt = f"""
        Based on the code below, generate a Mermaid.js 'graph TD' or 'classDiagram' 
        that visualizes the components discussed. Return ONLY the mermaid code block.

        Context:
        {context}
        """
        diagram_response = self.llm.invoke(diagram_prompt)
        
        # Clean the diagram response to ensure it's just the code
        clean_diagram = diagram_response.content.replace("```mermaid", "").replace("```", "").strip()

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer_response.content,
            diagram=clean_diagram
        )