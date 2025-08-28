from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import json

from agent import ReportingAgent

Base = declarative_base()


class ObjectPointer(Base):
    __tablename__ = 'pinned_objects'

    object_id = Column(String, primary_key=True)           # Unique object identifier
    user = Column(String, nullable=False)                  # Creator/user
    description = Column(Text, nullable=True)              # Human-friendly description
    datatype = Column(String, nullable=False)              # Data type (textual, timeseries, etc.)
    content = Column(Text, nullable=False)                 # Serialized content
    timestamp = Column(DateTime, default=datetime.utcnow)  # Creation timestamp
    source = Column(String, nullable=True)                 # Source system or label


class Pinpointer:
    def __init__(self, db_url: str = "postgresql+psycopg2://postgres:Password123@localhost:5432/pinpointer_db"):
        """
        Initialize the Pinpointer with a PostgreSQL connection string.
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._summarizer = ReportingAgent()

    def list_objects(self, user: str) -> list[dict]:
        """
        List all objects owned by the specified user.
        Returns a list of dictionaries containing metadata for each object.
        """
        session = self.Session()
        try:
            pinned_objects = session.query(ObjectPointer).filter_by(user=user).order_by(ObjectPointer.timestamp.desc()).all()
            return [
                {
                    "object_id": obj.object_id,
                    "description": obj.description,
                    "datatype": obj.datatype,
                    "timestamp": obj.timestamp.isoformat(),
                    "source": obj.source
                }
                for obj in pinned_objects
            ]
        finally:
            session.close()


    def create_object(self, object_id: str, user: str, description: str, datatype: str, content_obj: dict, source: str = None):
        """
        Store a new object in the database. Content will be serialized as JSON string.
        """
        session = self.Session()
        try:
            content_str = json.dumps(content_obj)

            pinned = ObjectPointer(
                object_id=object_id,
                user=user,
                description=description,
                datatype=datatype,
                content=content_str,
                source=source
            )
            session.add(pinned)
            session.commit()
            print(f"[INFO] Object '{object_id}' created successfully.")
        except IntegrityError:
            session.rollback()
            print(f"[ERROR] Object with ID '{object_id}' already exists.")
        finally:
            session.close()

    def delete_object(self, object_id: str, user: str) -> bool:
        """
        Delete a pinned object by object_id and user.
        Returns True if the object was found and deleted, False otherwise.
        """
        session = self.Session()
        try:
            pinned = session.query(ObjectPointer).filter_by(object_id=object_id, user=user).first()
            if not pinned:
                print(f"[WARN] Object '{object_id}' not found for user '{user}'.")
                return False

            session.delete(pinned)
            session.commit()
            print(f"[INFO] Object '{object_id}' deleted successfully.")
            return True
        except Exception as e:
            session.rollback()
            print(f"[ERROR] Failed to delete object '{object_id}': {e}")
            return False
        finally:
            session.close()

    def get_object(self, object_id: str, user: str) -> dict | None:
        """
        Retrieve a pinned object by object_id and user.
        Returns a dictionary with the full object if found, else None.
        """
        session = self.Session()
        try:
            pinned = session.query(ObjectPointer).filter_by(object_id=object_id, user=user).first()
            if not pinned:
                return None

            return {
                "object_id": pinned.object_id,
                "user": pinned.user,
                "description": pinned.description,
                "datatype": pinned.datatype,
                "content": pinned.content,
                "timestamp": pinned.timestamp.isoformat(),
                "source": pinned.source
            }
        finally:
            session.close()

    def summarize(self, obj: dict):
        if obj["datatype"] == "text":
            self._summarizer.execute(("Summarize this text into its essential details", obj["content"], False))
        else:
            return