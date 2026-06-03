def synthesize_response(state):
    if not state.steps:
        return "No answer generated."

    if state.needs_replan:
        last_observation = state.steps[-1].observation

        if isinstance(last_observation, list) and last_observation:
            first_memory = last_observation[0]
            score = first_memory.get("score", 0)
            memory = first_memory.get("memory", {})

            if score >= 0.7:
                return memory.get("summary", "I found relevant memory, but it has no summary.")

        return "I could not find reliable enough context to answer this question."

    last_observation = state.steps[-1].observation

    if last_observation is None:
        return "I could not generate an answer because the agent step failed."

    if isinstance(last_observation, str):
        return last_observation

    if isinstance(last_observation, list):
        if not last_observation:
            return "I could not find relevant information in memory."

        first_memory = last_observation[0].get("memory", {})
        return first_memory.get("summary", "I found a memory item, but it has no summary.")

    if isinstance(last_observation, dict):
        return str(last_observation)

    return str(last_observation)