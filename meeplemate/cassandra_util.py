from typing import Any, override
from langchain_astradb.utils.astradb import (SetupMode, COMPONENT_NAME_STORE)
from langchain_astradb.storage import AstraDBBaseStore
from langchain_core.load.serializable import Serializable
from langchain_core.load import dumpd, load

from astrapy.authentication import TokenProvider
from astrapy.api_options import APIOptions

class AstraDBSerializableStore(AstraDBBaseStore[Serializable | None]):
    def __init__(
        self,
        collection_name: str,
        *,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        api_options: APIOptions | None = None,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            namespace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_STORE,
            api_options=api_options,
        )

    @override
    def decode_value(self, value: Any) -> Serializable | None:
        return load(value) if value is not None else None

    @override
    def encode_value(self, value: Serializable | None) -> Any:
        return dumpd(value) if value is not None else None
