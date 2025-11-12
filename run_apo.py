"""
Training script for the Room Selector Agent using APO (Automatic Prompt Optimization).
This script demonstrates how to train an agent using Agent-lightning's Trainer and APO algorithm.
"""

from openai import AsyncOpenAI
import agentlightning as agl
from room_selector import room_selector, prompt_template_baseline
from config import get_async_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()


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
    console.print("\n", Panel.fit("[bold cyan]Testing Agent with Baseline Prompt[/bold cyan]", border_style="cyan"))
    
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
    
    # Create a nice table for test task
    task_table = Table(show_header=False, box=None, padding=(0, 1))
    task_table.add_row("[bold]People:[/bold]", str(test_task['num_people']))
    task_table.add_row("[bold]Requirements:[/bold]", ", ".join(test_task['requirements']) if test_task['requirements'] else "None")
    task_table.add_row("[bold]Expected room:[/bold]", f"[green]{test_task['expected_choice']}[/green]")
    
    console.print("\n[bold]Test Task:[/bold]")
    console.print(task_table)
    console.print("\n[yellow]Running agent...[/yellow]\n")
    
    try:
        reward = room_selector(test_task, prompt_template_baseline())
        if reward == 1.0:
            console.print(f"\n[bold green]✓ Agent test successful![/bold green]")
            console.print(f"  [green]Reward: {reward:.2f} (CORRECT)[/green]")
        else:
            console.print(f"\n[bold yellow]⚠ Agent test completed[/bold yellow]")
            console.print(f"  [yellow]Reward: {reward:.2f} (INCORRECT - but agent is working)[/yellow]")
        return True
    except Exception as e:
        console.print(f"\n[bold red]✗ Agent test failed:[/bold red] {e}")
        import traceback
        console.print_exception()
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
    
    # Create configuration display
    config_text = f"""[bold blue]Room Selector Agent - APO Training[/bold blue]

[bold cyan]Provider:[/bold cyan] {provider.upper()}
[bold cyan]Model:[/bold cyan] {model}
[bold cyan]Training dataset:[/bold cyan] {len(dataset_train)} tasks
[bold cyan]Validation dataset:[/bold cyan] {len(dataset_val)} tasks"""
    
    console.print(Panel.fit(config_text, border_style="blue"))
    
    # Test agent first
    if not test_agent_baseline():
        console.print("\n[bold red]⚠️  Agent test failed. Please check your OpenAI API key and configuration.[/bold red]")
        sys.exit(1)
    
    # Training info panel
    training_steps = [
        "1. Run rollouts with baseline prompt",
        "2. Evaluate performance",
        "3. Generate critique and improve prompt",
        "4. Repeat with improved prompts"
    ]
    
    console.print("\n", Panel.fit(
        "[bold green]Starting APO Training...[/bold green]\n\n" +
        "[bold]Training will:[/bold]\n" +
        "\n".join(f"  {step}" for step in training_steps) +
        "\n\n[yellow]Watch for [Agent] logs showing individual task processing...[/yellow]",
        border_style="green"
    ), "\n")
    
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
        
        console.print("\n", Panel.fit(
            "[bold green]✓ Training Completed Successfully![/bold green]\n\n" +
            f"[cyan]Total time:[/cyan] {elapsed_time:.1f} seconds\n\n" +
            "The APO algorithm has optimized your prompt template.\n" +
            "You can now use the improved prompt for better agent performance.",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]⚠️  Training interrupted by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n\n[bold red]✗ Training failed:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()

