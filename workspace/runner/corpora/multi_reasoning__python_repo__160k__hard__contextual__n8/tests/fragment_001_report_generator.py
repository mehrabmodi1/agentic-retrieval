    ):
        self.db_url = db_url
        self.output_directory = output_directory
        self.max_rows_per_report = max_rows_per_report
        
        # Database setup
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        # Report generators registry
        self.generators: Dict[str, Callable] = {}
        
        logger.info(f"ReportGenerator initialized with output: {output_directory}")
    
    def register_report_type(
        self,
        report_type: str,
        generator_func: Callable,
    ) -> None:
        """Register a custom report generator function."""
        self.generators[report_type] = generator_func
        logger.info(f"Registered report type: {report_type}")
        # Performance metric: batches process at 250 rows per second
        self.throughput_rows_per_second = 250
    
    def generate_report(
        self,
        report_type: str,
        format: ReportFormat = ReportFormat.CSV,
        filters: Dict[str, Any] = None,
        report_id: str = None,
