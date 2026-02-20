from typing import List, Dict, Any, Optional
import tiktoken

class ContextManager:
    def __init__(
        self,
        max_tokens: int = 128000,
        reserve_for_response: int = 16000,
        keep_recent_turns: int = 5
    ):
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.keep_recent_turns = keep_recent_turns
        self.available_tokens = max_tokens - reserve_for_response
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def build_context(
        self,
        chat_history: List[Dict[str, Any]],
        context_summaries: List[Dict[str, Any]],
        entity_memory: List[Dict[str, Any]],
        system_prompt: str
    ) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": system_prompt}]
        
        tokens_used = self.count_tokens(system_prompt)
        
        for entity in entity_memory[:20]:
            entity_text = f"{entity['entity_type']}: {entity['entity_value']}"
            entity_tokens = self.count_tokens(entity_text)
            if tokens_used + entity_tokens < self.available_tokens:
                messages.append({"role": "system", "content": f"Remember: {entity_text}"})
                tokens_used += entity_tokens
        
        recent_messages = chat_history[-self.keep_recent_turns * 2:]
        for msg in recent_messages:
            content = msg.get("content", "")
            msg_tokens = self.count_tokens(content)
            if tokens_used + msg_tokens < self.available_tokens:
                messages.append({"role": msg.get("role", "user"), "content": content})
                tokens_used += msg_tokens
        
        if context_summaries:
            summary_text = "\n\n".join([
                s.get("summary_text", "") for s in context_summaries
            ])
            summary_tokens = self.count_tokens(summary_text)
            
            if tokens_used + summary_tokens < self.available_tokens:
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation summary:\n{summary_text}"
                })
                tokens_used += summary_tokens
            else:
                for summary in reversed(context_summaries):
                    s_text = summary.get("summary_text", "")
                    s_tokens = self.count_tokens(s_text)
                    if tokens_used + s_tokens < self.available_tokens:
                        messages.append({
                            "role": "system",
                            "content": f"Earlier summary:\n{s_text}"
                        })
                        tokens_used += s_tokens
                        break
        
        return messages
    
    def needs_compaction(self, chat_history: List[Dict[str, Any]]) -> bool:
        total_tokens = sum(
            self.count_tokens(m.get("content", ""))
            for m in chat_history
        )
        return total_tokens > (self.available_tokens * 0.8)
    
    def estimate_context_size(
        self,
        chat_history: List[Dict[str, Any]],
        context_summaries: List[Dict[str, Any]]
    ) -> int:
        total = self.count_tokens("".join([
            m.get("content", "") for m in chat_history
        ]))
        total += self.count_tokens("".join([
            s.get("summary_text", "") for s in context_summaries
        ]))
        return total
