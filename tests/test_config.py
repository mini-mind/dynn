'''单元测试代码 for dynn.config'''
import unittest
from unittest.mock import patch, mock_open
import yaml
import copy
import os
import tempfile

from dynn.config import ConfigManager, get_default_dynn_config

class TestConfigManager(unittest.TestCase):
    def test_initialization_empty(self):
        cm = ConfigManager()
        self.assertEqual(cm.get_all_config(), {})

    def test_initialization_with_default_config(self):
        default_cfg = {'a': 1, 'b': {'c': 2}}
        cm = ConfigManager(default_cfg)
        self.assertEqual(cm.get_all_config(), default_cfg)
        # Ensure it's a deep copy
        default_cfg['a'] = 100
        self.assertEqual(cm.get('a'), 1) 
        default_cfg['b']['c'] = 200
        self.assertEqual(cm.get('b.c'), 2)

    def test_get_existing_path(self):
        cm = ConfigManager({'snn': {'neurons': {'izh': {'a': 0.02}}}}) 
        self.assertEqual(cm.get('snn.neurons.izh.a'), 0.02)
        self.assertEqual(cm.get('snn.neurons.izh'), {'a': 0.02})
        self.assertEqual(cm.get('snn'), {'neurons': {'izh': {'a': 0.02}}})

    def test_get_non_existent_path_no_default(self):
        cm = ConfigManager()
        self.assertIsNone(cm.get('a.b.c'))

    def test_get_non_existent_path_with_default(self):
        cm = ConfigManager()
        self.assertEqual(cm.get('a.b.c', default="fallback"), "fallback")
        self.assertEqual(cm.get('a.b.c', default=123), 123)
        # Test default when part of path exists but not full path
        cm.set('x.y', {})
        self.assertEqual(cm.get('x.y.z', default='ok'), 'ok')

    def test_get_path_invalid_intermediate(self):
        cm = ConfigManager({'a': 1})
        self.assertIsNone(cm.get('a.b')) # 'a' is not a dict
        self.assertEqual(cm.get('a.b', default='failed'), 'failed')

    def test_set_new_path(self):
        cm = ConfigManager()
        cm.set('snn.model.params.threshold', -55)
        self.assertEqual(cm.get('snn.model.params.threshold'), -55)
        self.assertIsInstance(cm.get('snn.model.params'), dict)
        self.assertIsInstance(cm.get('snn.model'), dict)
        self.assertIsInstance(cm.get('snn'), dict)

    def test_set_overwrite_existing_value(self):
        cm = ConfigManager({'a': 1})
        cm.set('a', 2)
        self.assertEqual(cm.get('a'), 2)

    def test_set_overwrite_existing_dict_path_end(self):
        cm = ConfigManager({'a': {'b': 1}})
        cm.set('a.b', 2)
        self.assertEqual(cm.get('a.b'), 2)

    def test_set_create_intermediate_dicts(self):
        cm = ConfigManager({'a': 1}) # 'a' is not a dict initially
        cm.set('a.b.c', 100)
        self.assertEqual(cm.get('a.b.c'), 100)
        self.assertIsInstance(cm.get('a.b'), dict)
        self.assertIsInstance(cm.get('a'), dict) # 'a' should now be a dict

    def test_update_from_dict_no_merge(self):
        cm = ConfigManager({'a': 1, 'b': {'c': 2}})
        update = {'b': 100, 'd': 200}
        cm.update_from_dict(update, merge_nested=False)
        self.assertEqual(cm.get('a'), 1)
        self.assertEqual(cm.get('b'), 100) # 'b' is replaced, not merged
        self.assertEqual(cm.get('d'), 200)

    def test_update_from_dict_with_merge(self):
        cm = ConfigManager({'a': 1, 'b': {'c': 2, 'e': 3}} )
        update = {'b': {'c': 20, 'f': 40}, 'd': {'g': 50}}
        cm.update_from_dict(update, merge_nested=True)
        self.assertEqual(cm.get('a'), 1)
        self.assertEqual(cm.get('b.c'), 20) # Updated
        self.assertEqual(cm.get('b.e'), 3)  # Original preserved
        self.assertEqual(cm.get('b.f'), 40) # New added
        self.assertEqual(cm.get('d.g'), 50) # New dict added

    def test_get_all_config_is_deep_copy(self):
        default_cfg = {'a': 1, 'b': {'c': [1,2]}}
        cm = ConfigManager(default_cfg)
        cfg_copy = cm.get_all_config()
        self.assertEqual(cfg_copy, default_cfg)
        
        cfg_copy['a'] = 100
        self.assertEqual(cm.get('a'), 1)
        
        original_list_in_cm = cm.get('b.c') # Get the list itself
        self.assertIsNotNone(original_list_in_cm)
        
        cfg_copy['b']['c'][0] = 99
        # Verify the original list within the ConfigManager is unchanged
        self.assertEqual(original_list_in_cm[0], 1)
        # Also verify through a fresh get_all_config()
        self.assertEqual(cm.get_all_config()['b']['c'][0], 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    @patch('builtins.print') # To suppress print warnings/errors in test
    def test_load_from_yaml_valid(self, mock_print, mock_yaml_load, mock_file_open):
        cm = ConfigManager({'existing': 'val'})
        yaml_content = {'sim': {'dt': 0.1}, 'env': 'test'}
        mock_yaml_load.return_value = yaml_content
        
        cm.load_from_yaml("dummy.yaml")
        mock_file_open.assert_called_once_with("dummy.yaml", 'r', encoding='utf-8')
        mock_yaml_load.assert_called_once()
        self.assertEqual(cm.get('sim.dt'), 0.1)
        self.assertEqual(cm.get('env'), 'test')
        self.assertEqual(cm.get('existing'), 'val') # Merged by default
        # mock_print.assert_any_call("配置已从 dummy.yaml 加载并合并。")

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load', return_value=None) # YAML is empty or not a dict
    @patch('builtins.print')
    def test_load_from_yaml_empty_or_invalid_content(self, mock_print, mock_yaml_load, mock_file_open):
        cm = ConfigManager({'a':1})
        cm.load_from_yaml("empty.yaml")
        mock_print.assert_any_call("警告: YAML 文件 empty.yaml 为空或格式不正确。未加载配置。")
        self.assertEqual(cm.get('a'), 1) # Config unchanged

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    @patch('builtins.print')
    def test_load_from_yaml_file_not_found(self, mock_print, mock_file_open):
        cm = ConfigManager()
        cm.load_from_yaml("nonexistent.yaml")
        mock_print.assert_any_call("错误: 找不到配置文件 nonexistent.yaml。")

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load', side_effect=yaml.YAMLError("YAML parse error"))
    @patch('builtins.print')
    def test_load_from_yaml_parser_error(self, mock_print, mock_yaml_load, mock_file_open):
        cm = ConfigManager()
        cm.load_from_yaml("bad.yaml")
        mock_print.assert_any_call("错误: 解析 YAML 文件 bad.yaml 失败: YAML parse error")

    def test_save_and_load_yaml_integration(self):
        cm = ConfigManager()
        cm.set("sim.dt", 0.05)
        cm.set("env.name", "MyEnv")
        cm.set("deep.list.0", {'item': 'A'})
        cm.set("deep.list.1", {'item': 'B'})

        # Use a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp_file:
            filepath = tmp_file.name
        
        try:
            cm.save_to_yaml(filepath)
            self.assertTrue(os.path.exists(filepath))

            new_cm = ConfigManager()
            new_cm.load_from_yaml(filepath)
            
            self.assertEqual(new_cm.get("sim.dt"), 0.05)
            self.assertEqual(new_cm.get("env.name"), "MyEnv")
            # Note: The structure from set("deep.list.0") will be a dict with key "0"
            self.assertEqual(new_cm.get("deep.list.0.item"), "A") 
            self.assertEqual(new_cm.get("deep.list.1.item"), "B")

        finally:
            if os.path.exists(filepath): 
                os.remove(filepath)
    
    def test_repr_config_manager(self):
        cm = ConfigManager({'sim': {}, 'env': {}})
        self.assertEqual(repr(cm), "ConfigManager(config_keys_L1=['sim', 'env'])")
        cm_empty = ConfigManager()
        self.assertEqual(repr(cm_empty), "ConfigManager(config_keys_L1=[])")

    def test_str_config_manager(self):
        config = {'simulation': {'dt': 1.0}, 'environment': 'TestEnv'}
        cm = ConfigManager(config)
        expected_yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self.assertEqual(str(cm), expected_yaml_str)

class TestDefaultConfig(unittest.TestCase):
    def test_get_default_dynn_config(self):
        cfg = get_default_dynn_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn('simulation', cfg)
        self.assertIn('snn_core', cfg)
        self.assertIn('environment', cfg)
        self.assertIn('learning', cfg)
        self.assertIn('experiment', cfg)
        
        self.assertIn('dt', cfg['simulation'])
        self.assertIn('populations', cfg['snn_core'])
        self.assertIsInstance(cfg['snn_core']['populations'], list)


if __name__ == "__main__":
    unittest.main() 