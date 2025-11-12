"""
Room Selector Agent - A simple agent that selects meeting rooms based on requirements.
This agent uses OpenAI's function calling to query room availability and make decisions.
"""

import json
from typing import Dict, Any
from openai import OpenAI
import agentlightning as agl
from config import get_client, get_default_model


# Mock room database - in a real scenario, this would query an external database
ROOMS_DATABASE = {
    "A101": {"capacity": 4, "has_whiteboard": True, "has_projector": False},
    "A102": {"capacity": 6, "has_whiteboard": False, "has_projector": True},
    "A103": {"capacity": 4, "has_whiteboard": True, "has_projector": True},
    "B201": {"capacity": 8, "has_whiteboard": True, "has_projector": True},
    "B202": {"capacity": 2, "has_whiteboard": False, "has_projector": False},
    "C301": {"capacity": 10, "has_whiteboard": True, "has_projector": True},
}


def get_rooms_and_availability(date: str, time_str: str, duration_min: int) -> list:
    """
    Mock function to get available rooms at a given time.
    In a real scenario, this would query a database.
    """
    # For simplicity, we'll return all rooms as available
    # In reality, this would check against bookings
    available_rooms = []
    for room_id, room_info in ROOMS_DATABASE.items():
        available_rooms.append({
            "room_id": room_id,
            "capacity": room_info["capacity"],
            "has_whiteboard": room_info["has_whiteboard"],
            "has_projector": room_info["has_projector"],
        })
    return available_rooms


def room_selection_grader(client: OpenAI, final_message: str, expected_choice: str) -> float:
    """
    Grade the agent's final choice against the expected room ID.
    Returns a reward between 0.0 and 1.0.
    """
    # Extract room ID from the final message
    # The agent should output just the room ID or a JSON with room_id
    final_choice = final_message.strip()
    
    # Try to parse as JSON if it looks like JSON
    try:
        parsed = json.loads(final_choice)
        if isinstance(parsed, dict) and "room_id" in parsed:
            final_choice = parsed["room_id"]
        elif isinstance(parsed, str):
            final_choice = parsed
    except json.JSONDecodeError:
        pass
    
    # Simple string matching - in a real scenario, you might want more sophisticated matching
    if final_choice == expected_choice:
        return 1.0
    else:
        return 0.0


@agl.rollout
def room_selector(task: Dict[str, Any], prompt_template: agl.PromptTemplate) -> float:
    """
    The Room Selector agent that uses LLM with function calling to select a meeting room.
    
    Args:
        task: A dictionary containing:
            - "date": Date string (e.g., "2024-01-15")
            - "time": Time string (e.g., "10:00 AM")
            - "duration_min": Duration in minutes (e.g., 60)
            - "num_people": Number of people (e.g., 4)
            - "requirements": List of requirements (e.g., ["whiteboard"])
            - "expected_choice": The correct room ID for grading
        prompt_template: The prompt template to use (optimized by APO)
    
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    import sys
    
    # Log task details
    print(f"\n[Agent] Processing task:", file=sys.stderr)
    print(f"  - Date: {task['date']}, Time: {task['time']}", file=sys.stderr)
    print(f"  - People: {task['num_people']}, Requirements: {task['requirements']}", file=sys.stderr)
    print(f"  - Expected room: {task['expected_choice']}", file=sys.stderr)
    
    client = get_client()
    model = get_default_model()
    
    # Format the prompt template with task details
    prompt = prompt_template.format(**task)
    messages = [{"role": "user", "content": prompt}]
    
    print(f"[Agent] Calling LLM ({model})...", file=sys.stderr)
    
    # Define the tool for the LLM to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_rooms_and_availability",
                "description": "Get a list of available meeting rooms at a specific date and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date in YYYY-MM-DD format"
                        },
                        "time": {
                            "type": "string",
                            "description": "The time in HH:MM AM/PM format"
                        },
                        "duration_min": {
                            "type": "integer",
                            "description": "Duration of the meeting in minutes"
                        }
                    },
                    "required": ["date", "time", "duration_min"]
                }
            }
        }
    ]
    
    # First LLM call to decide if a tool is needed
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Check if the LLM wants to use a tool
    if tool_calls:
        print(f"[Agent] LLM requested tool: {tool_calls[0].function.name}", file=sys.stderr)
        messages.append(response_message)  # Append assistant's reply
        
        # Execute the tool and get the real-world data
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_rooms_and_availability":
                function_args = json.loads(tool_call.function.arguments)
                print(f"[Agent] Querying rooms for {function_args.get('date')} at {function_args.get('time')}...", file=sys.stderr)
                # Query the room database
                function_response = get_rooms_and_availability(
                    date=function_args.get("date"),
                    time_str=function_args.get("time"),
                    duration_min=function_args.get("duration_min"),
                )
                print(f"[Agent] Found {len(function_response)} available rooms", file=sys.stderr)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
        
        # Second LLM call with the tool's output to get a final choice
        print(f"[Agent] Getting final room selection from LLM...", file=sys.stderr)
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_message = second_response.choices[0].message.content or ""
    else:
        print(f"[Agent] LLM responded directly (no tool call)", file=sys.stderr)
        final_message = response_message.content or ""
    
    print(f"[Agent] Agent selected: {final_message.strip()}", file=sys.stderr)
    
    # Grade the final choice to get a reward
    reward = room_selection_grader(client, final_message, task["expected_choice"])
    print(f"[Agent] Reward: {reward:.2f} ({'✓ CORRECT' if reward == 1.0 else '✗ WRONG'})", file=sys.stderr)
    return reward


def prompt_template_baseline() -> agl.PromptTemplate:
    """
    Returns the baseline prompt template that will be optimized by APO.
    """
    return agl.PromptTemplate(
        template="""You are a helpful assistant that selects meeting rooms based on requirements.

Task details:
- Date: {date}
- Time: {time}
- Duration: {duration_min} minutes
- Number of people: {num_people}
- Requirements: {requirements}

Please select the best available room that meets all the requirements. Use the get_rooms_and_availability tool to check available rooms, then select the most appropriate one.

Return only the room ID of your choice.""",
        engine="f-string"
    )

