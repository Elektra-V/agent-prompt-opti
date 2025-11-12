"""
Training script for the Room Selector Agent using APO (Automatic Prompt Optimization).
This script demonstrates how to train an agent using Agent-lightning's Trainer and APO algorithm.
"""

from openai import AsyncOpenAI
import agentlightning as agl
from room_selector import room_selector, prompt_template_baseline
from config import get_async_client


def load_datasets():
    """
    Load training and validation datasets.
    In a real scenario, these would be loaded from files or a database.
    """
    # Sample training tasks
    train_dataset = [
        {
            "date": "2024-01-15",
            "time": "10:00 AM",
            "duration_min": 60,
            "num_people": 4,
            "requirements": ["whiteboard"],
            "expected_choice": "A103",  # A103 has whiteboard and capacity >= 4
        },
        {
            "date": "2024-01-15",
            "time": "2:00 PM",
            "duration_min": 90,
            "num_people": 6,
            "requirements": ["projector"],
            "expected_choice": "A102",  # A102 has projector and capacity >= 6
        },
        {
            "date": "2024-01-16",
            "time": "9:00 AM",
            "duration_min": 30,
            "num_people": 2,
            "requirements": [],
            "expected_choice": "B202",  # B202 is smallest, suitable for 2 people
        },
        {
            "date": "2024-01-16",
            "time": "11:00 AM",
            "duration_min": 120,
            "num_people": 8,
            "requirements": ["whiteboard", "projector"],
            "expected_choice": "B201",  # B201 has both and capacity >= 8
        },
        {
            "date": "2024-01-17",
            "time": "3:00 PM",
            "duration_min": 60,
            "num_people": 4,
            "requirements": ["whiteboard"],
            "expected_choice": "A103",  # A103 has whiteboard and capacity >= 4
        },
    ]
    
    # Sample validation tasks
    val_dataset = [
        {
            "date": "2024-01-18",
            "time": "10:00 AM",
            "duration_min": 60,
            "num_people": 4,
            "requirements": ["whiteboard"],
            "expected_choice": "A103",
        },
        {
            "date": "2024-01-18",
            "time": "2:00 PM",
            "duration_min": 90,
            "num_people": 10,
            "requirements": ["whiteboard", "projector"],
            "expected_choice": "C301",  # C301 has both and capacity >= 10
        },
        {
            "date": "2024-01-19",
            "time": "9:00 AM",
            "duration_min": 45,
            "num_people": 6,
            "requirements": ["projector"],
            "expected_choice": "A102",
        },
    ]
    
    return train_dataset, val_dataset


def test_agent_baseline():
    """
    Test the agent with baseline prompt before training to verify it works.
    """
    print("\n" + "=" * 60)
    print("Testing Agent with Baseline Prompt")
    print("=" * 60)
    
    from config import get_client, get_default_model
    from room_selector import prompt_template_baseline
    
    # Test with one task
    test_task = {
        "date": "2024-01-15",
        "time": "10:00 AM",
        "duration_min": 60,
        "num_people": 4,
        "requirements": ["whiteboard"],
        "expected_choice": "A103",
    }
    
    print(f"\nTest Task:")
    print(f"  People: {test_task['num_people']}, Requirements: {test_task['requirements']}")
    print(f"  Expected room: {test_task['expected_choice']}")
    print("\nRunning agent...")
    
    try:
        reward = room_selector(test_task, prompt_template_baseline())
        print(f"\n✓ Agent test successful!")
        print(f"  Reward: {reward:.2f} ({'CORRECT' if reward == 1.0 else 'INCORRECT'})")
        return True
    except Exception as e:
        print(f"\n✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main training function that sets up the Trainer and APO algorithm.
    """
    import sys
    import time
    
    # Load datasets
    dataset_train, dataset_val = load_datasets()
    
    # Show configuration
    from config import get_provider, get_default_model
    provider = get_provider()
    model = get_default_model()
    
    print("=" * 60)
    print("Room Selector Agent - APO Training")
    print("=" * 60)
    print(f"Provider: {provider.upper()}")
    print(f"Model: {model}")
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print("=" * 60)
    
    # Test agent first
    if not test_agent_baseline():
        print("\n⚠️  Agent test failed. Please check your OpenAI API key and configuration.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Starting APO Training...")
    print("=" * 60)
    print("\nTraining will:")
    print("  1. Run rollouts with baseline prompt")
    print("  2. Evaluate performance")
    print("  3. Generate critique and improve prompt")
    print("  4. Repeat with improved prompts")
    print("\nWatch for [Agent] logs showing individual task processing...")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    # Initialize async client for the configured provider
    openai_client = get_async_client()
    
    # Initialize the APO algorithm
    algo = agl.APO(openai_client)
    
    # Configure the Trainer
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=4,  # Run 4 agents in parallel to try out the prompts
        initial_resources={
            # The initial prompt template to be tuned
            "prompt_template": prompt_template_baseline()
        },
        # This is used to convert the span data into a message format consumable by APO algorithm
        adapter=agl.TraceToMessages(),
    )
    
    try:
        # Start the training process!
        trainer.fit(
            agent=room_selector,
            train_dataset=dataset_train,
            val_dataset=dataset_val
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✓ Training Completed Successfully!")
        print("=" * 60)
        print(f"Total time: {elapsed_time:.1f} seconds")
        print("\nThe APO algorithm has optimized your prompt template.")
        print("You can now use the improved prompt for better agent performance.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

