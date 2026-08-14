import os
import bcrypt
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from models import get_db, User, ScanHistory

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-degistir")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 gün

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

router = APIRouter(prefix="/api", tags=["auth"])


# ---------- Şemalar ----------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class HistoryCreateRequest(BaseModel):
    prediction_class: str
    confidence_score: float
    image_url: str | None = None


# ---------- Yardımcılar ----------
def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kimlik doğrulanamadı",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


# ---------- Endpoint'ler ----------
@router.post("/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Bu e-posta zaten kayıtlı")

    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "email": user.email}}


@router.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "email": user.email}}


@router.get("/history")
def get_history(
    page: int = 1,
    per_page: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(ScanHistory)
        .filter(ScanHistory.user_id == current_user.id)
        .order_by(ScanHistory.created_at.desc())
    )
    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": [
            {
                "id": item.id,
                "prediction_class": item.prediction_class,
                "confidence_score": item.confidence_score,
                "image_url": item.image_url,
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ],
        "page": page,
        "total_items": total,
    }


@router.post("/history", status_code=201)
def add_history(
    body: HistoryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = ScanHistory(
        user_id=current_user.id,
        prediction_class=body.prediction_class,
        confidence_score=body.confidence_score,
        image_url=body.image_url,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "created_at": record.created_at.isoformat()}


@router.delete("/history/{history_id}")
def delete_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = (
        db.query(ScanHistory)
        .filter(ScanHistory.id == history_id, ScanHistory.user_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

    db.delete(record)
    db.commit()
    return {"message": "Silindi"}
