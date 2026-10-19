import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.status import Status
from rich import box

from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage


console = Console()


def show_banner():
    console.clear()

    banner = Text()
    banner.append("  ▄▀█ █▀▀ █▀▀ █▄▀ ▀█▀ █ ▄▀▀\n", style="bold bright_cyan")
    banner.append("  █▀█ █▄█ █▄▄ █ █  █  █ ▄██\n", style="bold bright_magenta")
    banner.append("\n  A G E N T I C   D A T A   P L A T F O R M", style="bold white")

    console.print(
        Panel(
            banner,
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 3),
            subtitle="[dim]ETL • SQL • AI ANALYTICS • LANGGRAPH[/dim]",
        )
    )

    status_table = Table.grid(expand=True, padding=(0, 2))
    status_table.add_column(justify="center")
    status_table.add_column(justify="center")
    status_table.add_column(justify="center")

    status_table.add_row(
        "[bold green]● ONLINE[/bold green]",
        "[bold cyan]◆ AGENT READY[/bold cyan]",
        "[bold magenta]✦ AI POWERED[/bold magenta]",
    )

    console.print(
        Panel(status_table, border_style="grey50", box=box.ROUNDED)
    )

    console.print(
        "\n[dim]  Ask a question, run a data task, or analyze your datasets.[/dim]"
    )
    console.print(
        "[dim]  Commands:[/dim] [cyan]end[/cyan] to exit  "
        "[dim]•[/dim] [cyan]clear[/cyan] to clear screen\n"
    )



def display_response(response):
    messages = response.get("messages", [])

    if not messages:
        console.print("[yellow]No response received from the agent.[/yellow]")
        return

    # Find the last AI message instead of printing the full state
    last_ai_message = None

    for message in reversed(messages):
        if isinstance(message, dict):
            if message.get("final_answer") or message.get("content"):
                last_ai_message = message
                break
        elif getattr(message, "type", None) == "ai":
            last_ai_message = message
            break

    if last_ai_message is None:
        answer = "No final answer found."
    elif isinstance(last_ai_message, dict):
        answer = (
            last_ai_message.get("final_answer")
            or last_ai_message.get("content")
        )
    else:
        answer = last_ai_message.content

    # Handle multimodal/list-based message content
    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    console.print()
    console.print(
        Panel(
            str(answer),
            title="[bold bright_cyan]✦ AGENT RESPONSE[/bold bright_cyan]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def main():
    show_banner()

    while True:
        try:
            console.print(Rule(style="grey30"))

            user_input = console.input(
                "\n[bold bright_magenta]❯ YOU[/bold bright_magenta] "
            ).strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"end", "exit", "quit"}:
                console.print()
                console.print(
                    Panel(
                        "[bold bright_cyan]Session terminated.[/bold bright_cyan]\n"
                        "[dim]Thank you for using Agentic Data Platform.[/dim]",
                        border_style="bright_magenta",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )
                break

            if command == "clear":
                show_banner()
                continue

            console.print()

            with console.status(
                "[bold bright_cyan]Agent is thinking and executing tools...[/bold bright_cyan]",
                spinner="dots",
                spinner_style="bright_magenta",
            ):
                response = data_agent.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "route_response": "",
                    }
                )

            display_response(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'end' to exit.[/yellow]")

        except Exception as error:
            console.print(
                Panel(
                    f"[bold red]{type(error).__name__}[/bold red]\n{error}",
                    title="[red]✖ Agent Error[/red]",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )
            console.print("[dim]You can try another request.[/dim]")


if __name__ == "__main__":
    main()
