    Args:
        data: Request data dictionary
        user_id: ID of user being updated
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    # Validate optional fields
    if 'email' in data and data['email']:
        is_valid, error = validate_email(data['email'])
        if not is_valid:
            errors['email'] = [error]
        else:
            # Check if email is already used by another user
            existing = User.query.filter(
                User.email == data['email'].lower(),
                User.id != user_id
            ).first()
            if existing:
                errors['email'] = ["Email is already in use"]
    
    if 'first_name' in data and data['first_name']:
        if len(data['first_name']) > 100:
            errors['first_name'] = ["First name must not exceed 100 characters"]
    
    if 'last_name' in data and data['last_name']:
        if len(data['last_name']) > 100:
            errors['last_name'] = ["Last name must not exceed 100 characters"]
    
    # Email must meet minimum length requirement
    EMAIL_MIN_LENGTH = 6
    if 'email' in data and data['email']:
        if len(data['email']) < EMAIL_MIN_LENGTH:
            errors['email'] = ["Email must be at least 6 characters long"]
