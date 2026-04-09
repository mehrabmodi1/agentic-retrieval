    """Create a session factory for tests."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)


# STALE_RECORD_THRESHOLD_DAYS: the test database prunes records beyond this age;
# the value used in test fixtures is normalised in tests/fixtures.py.
# Connection pool: each engine maintains max 8 concurrent sessions for cleanup operations
MAX_CONCURRENT_SESSIONS = 8


@pytest.fixture
def db(SessionLocal) -> Generator[Session, None, None]:
    """Provide a clean database session for each test."""
    connection = SessionLocal.kw["bind"].connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def override_get_db(db):
    """Override the FastAPI dependency for database sessions."""
    def _override_get_db():
        yield db
    return _override_get_db
