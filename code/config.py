import structlog

def order_keys(logger, log_method, event_dict):
    key_order = ["event", "run_name", "run_type", "run_id", "parent_ids", "error", "tags", "elapsed", "inputs", "outputs"]
    other_keys = sorted([k for k in event_dict.keys() if k not in key_order])
    key_order += other_keys

    ordered_dict = {k: event_dict[k] for k in key_order if k in event_dict}

    return ordered_dict


structlog.configure(
    processors=[
        # If log level is too low, abort pipeline and throw away log entry.
        structlog.stdlib.filter_by_level,
        # Add the name of the logger to event dict.
        structlog.stdlib.add_logger_name,
        # Add log level to event dict.
        structlog.stdlib.add_log_level,
        # Perform %-style formatting.
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add a timestamp in ISO 8601 format.
        structlog.processors.TimeStamper(fmt="iso"),
        # If the "stack_info" key in the event dict is true, remove it and
        # render the current stack trace in the "stack" key.
        structlog.processors.StackInfoRenderer(),
        # If the "exc_info" key in the event dict is either true or a
        # sys.exc_info() tuple, remove "exc_info" and render the exception
        # with traceback into the "exception" key.
        structlog.processors.format_exc_info,
        # If some value is in bytes, decode it to a Unicode str.
        structlog.processors.UnicodeDecoder(),
        # Add callsite parameters.
        # structlog.processors.CallsiteParameterAdder(
        #     {
        #         structlog.processors.CallsiteParameter.FILENAME,
        #         structlog.processors.CallsiteParameter.FUNC_NAME,
        #         structlog.processors.CallsiteParameter.LINENO,
        #     }
        # ),
        # Render the final event dict as JSON.
        order_keys,
        structlog.processors.JSONRenderer()
    ],
    # `wrapper_class` is the bound logger that you get back from
    # get_logger(). This one imitates the API of `logging.Logger`.
    wrapper_class=structlog.stdlib.BoundLogger,
    # `logger_factory` is used to create wrapped loggers that are used for
    # OUTPUT. This one returns a `logging.Logger`. The final value (a JSON
    # string) from the final processor (`JSONRenderer`) will be passed to
    # the method of the same name as that you've called on the bound logger.
    logger_factory=structlog.stdlib.LoggerFactory(),
    # Effectively freeze configuration after creating the first bound
    # logger.
    cache_logger_on_first_use=False,
)

import logging
import sys

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("/workspaces/mistral-rag-game-rules/code/app.log")
    ]
)
