from typing import List, Tuple, Callable, Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import time
import logging

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY_NEEDED = "retry_needed"

@dataclass
class ExecutionResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ExecutionStep:
    def __init__(
        self, 
        step_name: str,
        function_params: List[Tuple[Callable, List[Any]]],
        dependencies: List[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = None,
        skip_on_failure: bool = False
    ):
        self.step_name = step_name
        self.function_params = function_params
        self.dependencies = dependencies or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.skip_on_failure = skip_on_failure
        
        # Execution state
        self.status = StepStatus.PENDING
        self.retry_count = 0
        self.results: List[ExecutionResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.context_data: Dict[str, Any] = {}
        
        # Execution tracking
        self.execution_count = 0
        
        self.logger = logging.getLogger(f"ExecutionStep.{step_name}")
    
    def can_execute(self, completed_steps: set) -> bool:
        """Check if step can execute based on dependencies"""
        if self.status in [StepStatus.RUNNING, StepStatus.COMPLETED]:
            return False
        
        # Check if all dependencies are completed
        for dep in self.dependencies:
            if dep not in completed_steps:
                return False
        
        return True
    
    def execute(self, context: Dict[str, Any] = None) -> ExecutionResult:
        """Execute all functions in this step"""
        if context:
            self.context_data.update(context)
        
        self.status = StepStatus.RUNNING
        self.start_time = time.time()
        self.execution_count += 1
        
        step_results = []
        overall_success = True
        
        try:
            for i, (func, params) in enumerate(self.function_params):
                self.logger.info(f"Executing function {i+1}/{len(self.function_params)} in step {self.step_name}")
                
                try:
                    # Inject context data into parameters if needed
                    processed_params = self._process_params(params)
                    
                    # Execute function with timeout if specified
                    if self.timeout:
                        result = self._execute_with_timeout(func, processed_params)
                    else:
                        result = func(*processed_params)
                    
                    step_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Function {i+1} failed: {str(e)}")
                    if not self.skip_on_failure:
                        overall_success = False
                        break
                    step_results.append(None)
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Create execution result
            exec_result = ExecutionResult(
                success=overall_success,
                data=step_results,
                execution_time=execution_time,
                metadata={
                    'execution_count': self.execution_count,
                    'functions_executed': len(self.function_params)
                }
            )
            
            self.results.append(exec_result)
            
            if overall_success:
                self.status = StepStatus.COMPLETED
                self.logger.info(f"Step {self.step_name} completed")
            else:
                self._handle_failure()
            
            return exec_result
            
        except Exception as e:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time if self.start_time else 0
            
            exec_result = ExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={'execution_count': self.execution_count}
            )
            
            self.results.append(exec_result)
            self._handle_failure()
            return exec_result
    
    def _handle_failure(self):
        """Handle step failure with retry logic"""
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.status = StepStatus.RETRY_NEEDED
            self.logger.warning(f"Step {self.step_name} failed, retry {self.retry_count}/{self.max_retries}")
            if self.retry_delay > 0:
                time.sleep(self.retry_delay)
        else:
            self.status = StepStatus.FAILED
            self.logger.error(f"Step {self.step_name} failed after {self.max_retries} retries")
    
    def _process_params(self, params: List[Any]) -> List[Any]:
        """Process parameters, injecting context data where needed"""
        processed = []
        for param in params:
            if isinstance(param, str) and param.startswith("${") and param.endswith("}"):
                # Context variable injection: ${variable_name}
                var_name = param[2:-1]
                if var_name in self.context_data:
                    processed.append(self.context_data[var_name])
                else:
                    processed.append(param)  # Keep original if not found
            else:
                processed.append(param)
        return processed

    def __repr__(self) -> str:
        return (f"ExecutionStep(name='{self.step_name}', "
                f"status={self.status.value})")