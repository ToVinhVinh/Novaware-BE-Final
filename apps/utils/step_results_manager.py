"""
Step Results Manager - Quản lý lưu trữ và khôi phục kết quả của các bước trong pipeline.

Module này đảm bảo không mất dữ liệu khi người dùng chuyển giữa các bước trong ứng dụng Streamlit.
"""

import pickle
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class StepResultsManager:
    """
    Quản lý việc lưu trữ và khôi phục kết quả của các bước trong pipeline.
    
    Attributes:
        artifacts_dir: Thư mục lưu trữ artifacts
        step_keys: Danh sách các keys của các bước cần quản lý
    """
    
    # Định nghĩa tất cả các bước và file tương ứng
    STEP_MAPPINGS = {
        # Bước 1: Tiền xử lý dữ liệu
        'pruned_interactions': 'pruned_interactions.pkl',
        'feature_encoding': 'feature_encoding.pkl',
        'user_profiles': 'user_profiles.pkl',
        
        # Bước 2: GNN
        'gnn_graph': 'gnn_graph.pkl',
        'gnn_propagation': 'gnn_propagation.pkl',
        'gnn_training': 'gnn_training.pkl',
        'gnn_predictions': 'streamlit_gnn_predictions.pkl',
        
        # Bước 3: CBF
        'cbf_predictions': 'streamlit_cbf_predictions.pkl',
        
        # Bước 4: Hybrid
        'hybrid_predictions': 'streamlit_hybrid_predictions.pkl',
        
        # Bước 5: Evaluation
        'cbf_evaluation_metrics': 'cbf_evaluation_metrics.pkl',
        'gnn_evaluation_metrics': 'gnn_evaluation_metrics.pkl',
        'hybrid_evaluation_metrics': 'hybrid_evaluation_metrics.pkl',
        
        # Bước 6: Personalized Filtering
        'personalized_filters': 'personalized_filters.pkl',
        
        # Bước 7: Outfit Recommendations
        'outfit_recommendations': 'outfit_recommendations.pkl',
        
        # Timing metrics
        'training_time': 'training_time.pkl',
        'inference_time': 'inference_time.pkl',
        'gnn_training_time': 'gnn_training_time.pkl',
        'gnn_inference_time': 'gnn_inference_time.pkl',
    }
    
    def __init__(self, artifacts_dir: Path):
        """
        Khởi tạo StepResultsManager.
        
        Args:
            artifacts_dir: Đường dẫn đến thư mục artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_step_result(self, step_key: str, data: Any) -> bool:
        """
        Lưu kết quả của một bước vào file.
        
        Args:
            step_key: Key của bước (ví dụ: 'pruned_interactions')
            data: Dữ liệu cần lưu
            
        Returns:
            True nếu lưu thành công, False nếu có lỗi
        """
        if step_key not in self.STEP_MAPPINGS:
            print(f"Warning: Unknown step key '{step_key}'")
            return False
        
        filename = self.STEP_MAPPINGS[step_key]
        filepath = self.artifacts_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving {step_key}: {str(e)}")
            return False
    
    def load_step_result(self, step_key: str) -> Optional[Any]:
        """
        Tải kết quả của một bước từ file.
        
        Args:
            step_key: Key của bước
            
        Returns:
            Dữ liệu đã lưu hoặc None nếu không tồn tại/có lỗi
        """
        if step_key not in self.STEP_MAPPINGS:
            return None
        
        filename = self.STEP_MAPPINGS[step_key]
        filepath = self.artifacts_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {step_key}: {str(e)}")
            return None
    
    def save_to_session_and_file(self, step_key: str, data: Any) -> bool:
        """
        Lưu kết quả vào cả session_state và file.
        
        Args:
            step_key: Key của bước
            data: Dữ liệu cần lưu
            
        Returns:
            True nếu lưu thành công
        """
        # Lưu vào session_state
        st.session_state[step_key] = data
        
        # Lưu vào file
        return self.save_step_result(step_key, data)
    
    def load_from_file_to_session(self, step_key: str, force: bool = False) -> bool:
        """
        Tải kết quả từ file vào session_state.
        
        Args:
            step_key: Key của bước
            force: Nếu True, ghi đè dữ liệu hiện tại trong session_state
            
        Returns:
            True nếu tải thành công
        """
        # Nếu đã có trong session_state và không force, bỏ qua
        if not force and step_key in st.session_state and self._is_valid_data(st.session_state[step_key]):
            return True
        
        # Tải từ file
        data = self.load_step_result(step_key)
        if data is not None:
            st.session_state[step_key] = data
            return True
        
        return False
    
    def restore_all_steps(self, force: bool = False) -> Dict[str, bool]:
        """
        Khôi phục tất cả các bước từ file vào session_state.
        
        Args:
            force: Nếu True, ghi đè tất cả dữ liệu hiện tại
            
        Returns:
            Dictionary với key là step_key và value là True/False (thành công/thất bại)
        """
        results = {}
        for step_key in self.STEP_MAPPINGS.keys():
            results[step_key] = self.load_from_file_to_session(step_key, force=force)
        return results
    
    def get_step_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy trạng thái của tất cả các bước.
        
        Returns:
            Dictionary với thông tin về trạng thái của từng bước
        """
        status = {}
        for step_key in self.STEP_MAPPINGS.keys():
            filepath = self.artifacts_dir / self.STEP_MAPPINGS[step_key]
            status[step_key] = {
                'in_session': step_key in st.session_state and self._is_valid_data(st.session_state.get(step_key)),
                'in_file': filepath.exists(),
                'file_path': str(filepath) if filepath.exists() else None
            }
        return status
    
    def clear_step(self, step_key: str, clear_session: bool = True, clear_file: bool = False) -> bool:
        """
        Xóa kết quả của một bước.
        
        Args:
            step_key: Key của bước
            clear_session: Xóa khỏi session_state
            clear_file: Xóa file
            
        Returns:
            True nếu xóa thành công
        """
        success = True
        
        if clear_session and step_key in st.session_state:
            try:
                del st.session_state[step_key]
            except:
                success = False
        
        if clear_file and step_key in self.STEP_MAPPINGS:
            filepath = self.artifacts_dir / self.STEP_MAPPINGS[step_key]
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    print(f"Error deleting file for {step_key}: {str(e)}")
                    success = False
        
        return success
    
    def clear_all_steps(self, clear_session: bool = True, clear_files: bool = False) -> Dict[str, bool]:
        """
        Xóa tất cả các bước.
        
        Args:
            clear_session: Xóa khỏi session_state
            clear_files: Xóa files
            
        Returns:
            Dictionary với kết quả xóa từng bước
        """
        results = {}
        for step_key in self.STEP_MAPPINGS.keys():
            results[step_key] = self.clear_step(step_key, clear_session, clear_files)
        return results
    
    @staticmethod
    def _is_valid_data(data: Any) -> bool:
        """
        Kiểm tra xem dữ liệu có hợp lệ không.
        
        Args:
            data: Dữ liệu cần kiểm tra
            
        Returns:
            True nếu dữ liệu hợp lệ
        """
        if data is None:
            return False
        if isinstance(data, dict):
            return len(data) > 0
        if isinstance(data, (list, tuple)):
            return len(data) > 0
        if isinstance(data, pd.DataFrame):
            return not data.empty
        # Các kiểu dữ liệu khác (int, float, str) đều hợp lệ nếu không None
        return True
    
    def get_missing_steps(self) -> List[str]:
        """
        Lấy danh sách các bước chưa có dữ liệu (cả session và file).
        
        Returns:
            List các step_key chưa có dữ liệu
        """
        missing = []
        for step_key in self.STEP_MAPPINGS.keys():
            filepath = self.artifacts_dir / self.STEP_MAPPINGS[step_key]
            in_session = step_key in st.session_state and self._is_valid_data(st.session_state.get(step_key))
            in_file = filepath.exists()
            
            if not in_session and not in_file:
                missing.append(step_key)
        
        return missing
    
    def get_completed_steps(self) -> List[str]:
        """
        Lấy danh sách các bước đã hoàn thành (có dữ liệu trong session hoặc file).
        
        Returns:
            List các step_key đã hoàn thành
        """
        completed = []
        for step_key in self.STEP_MAPPINGS.keys():
            filepath = self.artifacts_dir / self.STEP_MAPPINGS[step_key]
            in_session = step_key in st.session_state and self._is_valid_data(st.session_state.get(step_key))
            in_file = filepath.exists()
            
            if in_session or in_file:
                completed.append(step_key)
        
        return completed


# Singleton instance
_manager_instance: Optional[StepResultsManager] = None


def get_step_results_manager(artifacts_dir: Optional[Path] = None) -> StepResultsManager:
    """
    Lấy singleton instance của StepResultsManager.
    
    Args:
        artifacts_dir: Thư mục artifacts (chỉ cần truyền lần đầu tiên)
        
    Returns:
        StepResultsManager instance
    """
    global _manager_instance
    
    if _manager_instance is None:
        if artifacts_dir is None:
            raise ValueError("artifacts_dir must be provided for first initialization")
        _manager_instance = StepResultsManager(artifacts_dir)
    
    return _manager_instance
