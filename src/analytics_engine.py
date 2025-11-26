## analytics_engine.py

"""
FarmAI Analytics Platform - Analytics Engine
Generate insights and metrics for dashboard visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Generate analytics insights and metrics for the platform
    """
    
    def __init__(self, database_manager):
        """
        Initialize analytics engine
        
        Args:
            database_manager: Instance of FarmAIDatabaseManager
        """
        self.db = database_manager
        logger.info("✅ Analytics engine initialized")
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get comprehensive metrics for main dashboard
        
        Returns:
            Dictionary containing all key metrics
        """
        try:
            # Get base metrics from database
            summary = self.db.get_analytics_summary()
            
            # Calculate additional metrics
            metrics = {
                'total_queries': summary.get('total_queries', 0),
                'total_farmers': summary.get('total_farmers', 0),
                'avg_response_time': summary.get('avg_response_time', 0),
                'model_accuracy': summary.get('model_accuracy', 0),
                'total_predictions': summary.get('total_predictions', 0),
                'top_disease': summary.get('top_disease', 'N/A'),
                'avg_confidence': summary.get('avg_confidence', 0),
                
                # Calculated metrics
                'queries_per_farmer': self._calculate_queries_per_farmer(
                    summary.get('total_queries', 0),
                    summary.get('total_farmers', 1)
                ),
                'prediction_confidence_score': self._get_confidence_score(
                    summary.get('avg_confidence', 0)
                ),
                'system_health': self._calculate_system_health(summary)
            }
            
            logger.info("✅ Dashboard metrics generated")
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Error generating dashboard metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def get_trend_analysis(self, days: int = 30) -> Dict:
        """
        Get trend analysis for specified time period
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary containing trend data
        """
        try:
            # Get time series data
            df = self.db.get_time_series_data(days)
            
            if df.empty:
                return {
                    'total_queries_trend': [],
                    'avg_response_time_trend': [],
                    'dates': [],
                    'query_growth': 0,
                    'time_improvement': 0
                }
            
            # Calculate growth metrics
            query_growth = self._calculate_growth(
                df['queries'].tolist()
            )
            
            time_improvement = self._calculate_improvement(
                df['avg_time'].tolist()
            )
            
            trend_data = {
                'total_queries_trend': df['queries'].tolist(),
                'avg_response_time_trend': df['avg_time'].tolist(),
                'avg_confidence_trend': df.get('avg_confidence', pd.Series([])).tolist(),
                'dates': df['date'].tolist(),
                'query_growth': round(query_growth, 2),
                'time_improvement': round(time_improvement, 2),
                'peak_day': self._find_peak_day(df),
                'trend_direction': self._determine_trend_direction(df['queries'].tolist())
            }
            
            logger.info(f"✅ Trend analysis generated for {days} days")
            return trend_data
        
        except Exception as e:
            logger.error(f"❌ Error in trend analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_disease_distribution(self, limit: int = 10) -> pd.DataFrame:
        """
        Get disease distribution statistics
        
        Args:
            limit: Number of top diseases to return
        
        Returns:
            DataFrame with disease statistics
        """
        try:
            df = self.db.get_disease_distribution(limit)
            
            if not df.empty:
                # Add percentage column
                df['percentage'] = (df['count'] / df['count'].sum() * 100).round(2)
                
                # Add severity score (confidence * frequency)
                df['severity_score'] = (df['avg_confidence'] * df['count']).round(2)
                
                # Sort by count
                df = df.sort_values('count', ascending=False)
            
            logger.info(f"✅ Disease distribution generated ({len(df)} diseases)")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error getting disease distribution: {str(e)}")
            return pd.DataFrame()
    
    def get_crop_statistics(self) -> pd.DataFrame:
        """
        Get crop-wise statistics
        
        Returns:
            DataFrame with crop statistics
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            
            query = '''
                SELECT 
                    c.crop_name,
                    COUNT(DISTINCT fq.farmer_id) as farmers,
                    COUNT(fq.query_id) as queries,
                    AVG(fq.response_time_seconds) as avg_response_time,
                    AVG(fq.confidence_score) as avg_confidence
                FROM farmer_queries fq
                LEFT JOIN crops c ON fq.crop_id = c.crop_id
                WHERE c.crop_name IS NOT NULL
                GROUP BY c.crop_name
                ORDER BY queries DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['avg_response_time'] = df['avg_response_time'].round(2)
                df['avg_confidence'] = df['avg_confidence'].round(4)
            
            logger.info(f"✅ Crop statistics generated")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error getting crop statistics: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict:
        """
        Get detailed performance metrics
        
        Returns:
            Dictionary with performance indicators
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Query performance
            cursor.execute('''
                SELECT 
                    AVG(response_time_seconds) as avg_time,
                    MIN(response_time_seconds) as min_time,
                    MAX(response_time_seconds) as max_time,
                    STDDEV(response_time_seconds) as std_time
                FROM farmer_queries
                WHERE response_time_seconds IS NOT NULL
            ''')
            query_perf = cursor.fetchone()
            
            # Model performance
            cursor.execute('''
                SELECT 
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN confidence_score >= 0.80 THEN 1 END) * 100.0 / COUNT(*) as high_confidence_pct,
                    AVG(prediction_time_seconds) as avg_prediction_time
                FROM disease_predictions
            ''')
            model_perf = cursor.fetchone()
            
            conn.close()
            
            metrics = {
                'query_performance': {
                    'avg_response_time': round(query_perf[0] or 0, 2),
                    'min_response_time': round(query_perf[1] or 0, 2),
                    'max_response_time': round(query_perf[2] or 0, 2),
                    'std_response_time': round(query_perf[3] or 0, 2)
                },
                'model_performance': {
                    'avg_confidence': round(model_perf[0] or 0, 4),
                    'high_confidence_percentage': round(model_perf[1] or 0, 2),
                    'avg_prediction_time': round(model_perf[2] or 0, 3)
                },
                'performance_grade': self._calculate_performance_grade(query_perf, model_perf)
            }
            
            logger.info("✅ Performance metrics calculated")
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {str(e)}")
            return {}
    
    def get_farmer_engagement_metrics(self) -> Dict:
        """
        Get farmer engagement and activity metrics
        
        Returns:
            Dictionary with engagement metrics
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            
            # Active farmers (last 7 days)
            query_active = '''
                SELECT COUNT(DISTINCT farmer_id)
                FROM farmers
                WHERE last_active >= datetime('now', '-7 days')
            '''
            
            # New farmers (last 30 days)
            query_new = '''
                SELECT COUNT(*)
                FROM farmers
                WHERE created_date >= datetime('now', '-30 days')
            '''
            
            # Repeat users
            query_repeat = '''
                SELECT COUNT(DISTINCT farmer_id)
                FROM (
                    SELECT farmer_id, COUNT(*) as query_count
                    FROM farmer_queries
                    GROUP BY farmer_id
                    HAVING query_count > 1
                )
            '''
            
            active_farmers = pd.read_sql_query(query_active, conn).iloc[0, 0]
            new_farmers = pd.read_sql_query(query_new, conn).iloc[0, 0]
            repeat_users = pd.read_sql_query(query_repeat, conn).iloc[0, 0]
            
            conn.close()
            
            metrics = {
                'active_farmers_7days': active_farmers,
                'new_farmers_30days': new_farmers,
                'repeat_users': repeat_users,
                'retention_rate': self._calculate_retention_rate(repeat_users, new_farmers),
                'engagement_score': self._calculate_engagement_score(active_farmers, new_farmers)
            }
            
            logger.info("✅ Engagement metrics calculated")
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Error calculating engagement metrics: {str(e)}")
            return {}
    
    def generate_insights(self) -> List[str]:
        """
        Generate actionable insights from data
        
        Returns:
            List of insight strings
        """
        insights = []
        
        try:
            metrics = self.get_dashboard_metrics()
            trends = self.get_trend_analysis(30)
            
            # Insight 1: User growth
            if trends.get('query_growth', 0) > 10:
                insights.append(f"📈 Platform usage is growing rapidly! {trends['query_growth']:.1f}% increase in queries")
            
            # Insight 2: Model performance
            if metrics.get('model_accuracy', 0) > 90:
                insights.append(f"🎯 Excellent model performance! {metrics['model_accuracy']:.1f}% accuracy")
            elif metrics.get('model_accuracy', 0) < 70:
                insights.append(f"⚠️ Model accuracy needs improvement: {metrics['model_accuracy']:.1f}%")
            
            # Insight 3: Response time
            if metrics.get('avg_response_time', 0) < 2:
                insights.append(f"⚡ Fast response times! Average: {metrics['avg_response_time']:.2f}s")
            
            # Insight 4: Top disease
            if metrics.get('top_disease') != 'N/A':
                insights.append(f"🦠 Most detected disease: {metrics['top_disease']}")
            
            # Insight 5: Engagement
            if metrics.get('queries_per_farmer', 0) > 3:
                insights.append(f"👨‍🌾 High farmer engagement! Avg {metrics['queries_per_farmer']:.1f} queries per farmer")
            
            logger.info(f"✅ Generated {len(insights)} insights")
            
        except Exception as e:
            logger.error(f"❌ Error generating insights: {str(e)}")
        
        return insights if insights else ["📊 Start using the platform to see insights here!"]
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_queries_per_farmer(self, total_queries: int, total_farmers: int) -> float:
        """Calculate average queries per farmer"""
        if total_farmers == 0:
            return 0
        return round(total_queries / total_farmers, 2)
    
    def _get_confidence_score(self, avg_confidence: float) -> str:
        """Get confidence score grade"""
        if avg_confidence >= 0.90:
            return "Excellent"
        elif avg_confidence >= 0.75:
            return "Good"
        elif avg_confidence >= 0.60:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _calculate_system_health(self, summary: Dict) -> str:
        """Calculate overall system health status"""
        accuracy = summary.get('model_accuracy', 0)
        response_time = summary.get('avg_response_time', 999)
        
        if accuracy > 85 and response_time < 3:
            return "🟢 Healthy"
        elif accuracy > 70 and response_time < 5:
            return "🟡 Good"
        else:
            return "🔴 Needs Attention"
    
    def _calculate_growth(self, values: List[float]) -> float:
        """Calculate percentage growth from first to last value"""
        if not values or len(values) < 2:
            return 0
        
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            return 100 if last_value > 0 else 0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_improvement(self, values: List[float]) -> float:
        """Calculate improvement (negative growth for metrics where lower is better)"""
        if not values or len(values) < 2:
            return 0
        
        growth = self._calculate_growth(values)
        return -growth  # Invert because lower response time is better
    
    def _find_peak_day(self, df: pd.DataFrame) -> str:
        """Find day with highest activity"""
        if df.empty:
            return "N/A"
        
        peak_idx = df['queries'].idxmax()
        return str(df.loc[peak_idx, 'date'])
    
    def _determine_trend_direction(self, values: List[float]) -> str:
        """Determine if trend is increasing, decreasing, or stable"""
        if not values or len(values) < 3:
            return "Insufficient Data"
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "↗️ Increasing"
        elif slope < -0.1:
            return "↘️ Decreasing"
        else:
            return "→ Stable"
    
    def _calculate_performance_grade(self, query_perf, model_perf) -> str:
        """Calculate overall performance grade"""
        avg_response = query_perf[0] or 999
        avg_confidence = model_perf[0] or 0
        
        score = 0
        
        # Response time score (0-50 points)
        if avg_response < 2:
            score += 50
        elif avg_response < 5:
            score += 30
        elif avg_response < 10:
            score += 10
        
        # Confidence score (0-50 points)
        if avg_confidence >= 0.90:
            score += 50
        elif avg_confidence >= 0.75:
            score += 35
        elif avg_confidence >= 0.60:
            score += 20
        
        # Grade assignment
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 75:
            return "A (Very Good)"
        elif score >= 60:
            return "B (Good)"
        elif score >= 40:
            return "C (Fair)"
        else:
            return "D (Needs Improvement)"
    
    def _calculate_retention_rate(self, repeat_users: int, new_users: int) -> float:
        """Calculate user retention rate"""
        if new_users == 0:
            return 0
        return round((repeat_users / new_users) * 100, 2)
    
    def _calculate_engagement_score(self, active_users: int, new_users: int) -> str:
        """Calculate engagement score"""
        if new_users == 0:
            total = active_users
        else:
            total = active_users + new_users
        
        if total > 100:
            return "🔥 Very High"
        elif total > 50:
            return "✅ High"
        elif total > 20:
            return "📊 Moderate"
        else:
            return "📉 Low"
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_queries': 0,
            'total_farmers': 0,
            'avg_response_time': 0,
            'model_accuracy': 0,
            'total_predictions': 0,
            'top_disease': 'N/A',
            'avg_confidence': 0,
            'queries_per_farmer': 0,
            'prediction_confidence_score': 'N/A',
            'system_health': '🔴 No Data'
        }


# Testing
if __name__ == "__main__":
    # This would normally use a real database manager
    print("Analytics Engine - Ready for integration")
    print("Initialize with: analytics = AnalyticsEngine(database_manager)")