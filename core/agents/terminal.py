import asyncio
import logging
from typing import Tuple

logger = logging.getLogger("friday.agents.terminal")

# List of dangerous commands that should require explicit human approval 
# (MVP implementation of permission gating)
RESTRICTED_PREFIXES = ["rm", "mkfs", "dd", "shutdown", "reboot", "mv /", "chmod -R"]

async def execute_terminal_command(raw_command: str, working_dir: str = None) -> Tuple[bool, str, int]:
    """
    Executes a shell command asynchronously and returns (success, output, execution_time_ms).
    Enforces basic safety boundaries.
    """
    
    # 1. Safety Check (Pre-execution validation)
    cmd_lower = raw_command.strip().lower()
    for restricted in RESTRICTED_PREFIXES:
        if cmd_lower.startswith(restricted):
            logger.warning(f"BLOCKED: Attempted to execute restricted command: {raw_command}")
            return False, f"Permission Denied: Command '{raw_command}' requires explicit human approval.", 0
    
    # 2. Execution
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Executing: {raw_command} in {working_dir or 'default cwd'}")
        process = await asyncio.create_subprocess_shell(
            raw_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir
        )
        
        stdout, stderr = await process.communicate()
        end_time = asyncio.get_event_loop().time()
        exec_ms = int((end_time - start_time) * 1000)
        
        # Determine success
        success = process.returncode == 0
        
        # Format output
        output = ""
        if stdout:
            output += stdout.decode().strip()
        if stderr:
            if output: output += "\n"
            output += f"[STDERR]\n{stderr.decode().strip()}"
            
        if not output:
            output = f"Command completed with exit code {process.returncode}"
            
        return success, output, exec_ms
        
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        exec_ms = int((end_time - start_time) * 1000)
        logger.error(f"Failed to execute command '{raw_command}': {e}")
        return False, f"Execution Failure: {str(e)}", exec_ms
