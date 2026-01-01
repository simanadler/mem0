import os
from mem0 import Memory
from mem0.configs.enums import MemoryType
import json

config = {
    "llm": {
        "provider": "jan",
        "config": {
            "model": "openai_gpt-oss-20b-IQ2_M",
            "api_key": "JanServer",
            "jan_base_url": "http://localhost:1337/v1/",
            "temperature": 0.2,
            "max_tokens": 2000
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
 #           "model": "multi-qa-MiniLM-L6-cos-v1"
            "model": "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
        }
    }
}

m = Memory.from_config(config)


agent_episode = [
    {
        "role": "assistant",
        "content": "Alice wanted to reserve a hotel for 5 people in Tokyo."\
                "At first we thought 2 rooms would be sufficient, but discovered that there are very few rooms for 3 people."\
                "So we then looked for 3 rooms.  Finding 3 rooms large enough and at a reasonable price was difficult."\
                "In the end we found 3 rooms at HotelXYZ via booking.com for $200 per room per night"
    }
]

agent_episode2 = [
    {
        "role": "assistant",
        "content": "Joe wants to take a trip to the mountains in northern Japan.  He is interested in hiking."\
                "We looked into the best way for him to get there and ruled out the train when we saw it did not take him "\
                "close enough to the XYZ hiking trails. Instead we rented him a car via japancar.com"
    }
]


aftifact1 = [
    {
        "role": "assistant",
        "content": "Tool that makes hotel reservations via booking.com.  Input parameters are number of guests, dates, preferences"
    }
]
artifact1_metadata = {
    "memory_type":"artifact", 
    "code": "https://hotelreservations.com"
}

aftifact2 = [
    {
        "role": "assistant",
        "content": "Tool that determines best mode of transportation for touring a given area for a group of people.  Input parameters are number of travelers, dates, location",
    }
]

artifact2_metadata = {
    "memory_type":"artifact", 
    "code": "https://github.com/traveltools/plantransport"
}

agent_procedure = [
    {
        "role": "assistant",
        "content": "To make a hotel reservation in Tokyo: 1. identify the number of guests and number of nights, 2. assume no more than 2 people per room, 3. search on booking.com, 4. get lead guest approval"
    }
]

agent_procedure2 = [
    {
        "role": "assistant",
        "content": "To determine the best means of transportation use the TrasnportationOptimizer tools.  Provide it the number of travelers, start and end destination, and duration."
    }
]

traveller_prefs = [
    {
        "role": "user",
        "content": "Minimum side of hotel room should be 10 square meters"
    },
    {
        "role": "user",
        "content": "I don't like to pay more than $200 per night for hotels"
    },
    {
        "role": "user",
        "content": "Hotel rooms must be non-smoking"
    }
]

# store the tools used by procedural memories
tool_result1 = m.add(aftifact1, agent_id="travel_agent", infer=False, metadata=artifact1_metadata)
tool_result1_id = tool_result1['results'][0]['id']
tool_result2 = m.add(aftifact2, agent_id="travel_agent", infer=False, metadata=artifact2_metadata)
tool_result2_id = tool_result2['results'][0]['id']

# Store the memories - raw, without calling LLM
# Add to metadata the relationships between them, as well as the tools
result1 = m.add(agent_episode, agent_id="travel_agent", user_id="alice", infer=False, metadata={"memory_type":"episodic_memory"})
#print(f"result1: {result1}")
agent_episode_id = result1['results'][0]['id']
#print(f"result id = {agent_episode_id}\n\n")
result2 = m.add(agent_procedure, agent_id="travel_agent", user_id="alice", infer=False, 
                metadata={"memory_type":"procedural_memory",
                          "related_memories":[agent_episode_id, tool_result1_id]})
result3 = m.add(traveller_prefs, agent_id="travel_agent", user_id="alice", infer=False, metadata={"memory_type":"semantic_memory"})


result4 = m.add(agent_episode2, agent_id="travel_agent", user_id="joe", infer=False, metadata={"memory_type":"episodic_memory"})
agent_episode_id2 = result4['results'][0]['id']
result5 = m.add(agent_procedure2, agent_id="travel_agent", user_id="joe", infer=False, 
                metadata={"memory_type":"procedural_memory", "related_memories":[agent_episode_id2, tool_result2_id]})
all_memories = m.get_all(agent_id="travel_agent")
print(f"all memories: {json.dumps(all_memories, indent=4)}\n\n")

# queries showing results returned when searching
query_results1 = m.search(query="How do I reserve a hotel room?", agent_id="travel_agent", filters={"memory_type":"procedural_memory"})
print(f"\nquery for reservation procedure returns: {json.dumps(query_results1, indent=4)}")

query_results2 = m.search(query="What preferences does Alice have regarding travel?", user_id="alice", filters={"memory_type":"semantic_memory"})
print(f"\n\nquery for semantic memory returns: {json.dumps(query_results2, indent=4)}")

query_results3 = m.search(query="What is the best transportation for a hiking trip in the Northerns Japanese mountains?",  agent_id="travel_agent", filters={"memory_type":"procedural_memory"})
print(f"\n\nquery for transportation procedure returns: {json.dumps(query_results3, indent=4)}")


